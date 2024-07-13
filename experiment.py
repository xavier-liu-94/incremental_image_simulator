import os
os.environ["MODELSCOPE_CACHE"] = "/media/xavier/Samsumg/.cache/modelscope/hub"
import argparse
import hashlib
import itertools
import math
import os
import inspect
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint


from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami

from tqdm.auto import tqdm
from transformers import ViTModel, ViTConfig
from diffusers.models.autoencoders.vae import Encoder, Decoder, DiagonalGaussianDistribution
# from transformers.models.vit.modeling_vit import ViTEncoder, ViTEmbeddings

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn import functional as F

import torchvision.utils as vutil
import uuid

from pathlib import Path
from modelscope.msdatasets import MsDataset
from modelscope.hub.api import HubApi
api = HubApi()
api.login('f9dbdac5-f46c-426c-abe0-8a73bb628d96')

TRANS = transforms.Compose(
            [transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])]
        )

def denormalize(images):
        """
        Denormalize an image array to [0,1].
        """
        return (images / 2 + 0.5).clamp(0, 1)

def postprocess(batch_img):
    return torch.stack(
            [denormalize(batch_img[i]) for i in range(batch_img.shape[0])]
        )


class MsImageDataset(Dataset):

    def __init__(self, ms: MsDataset) -> None:
        super().__init__()
        self.ms = ms
        self.image_transforms = transforms.Compose(
            [transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(256),
            transforms.ColorJitter(0.2, 0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])]
        )
    
    def __getitem__(self, index):
        instance_image = Image.open(self.ms[index]['image:FILE'])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        return self.image_transforms(instance_image)
    
    def __len__(self):
        return len(self.ms)


def deploy(batch_patch, patch_size=32):
    # batch*patch_size*patch_size, dim, 1, 1 -> batch, dim, patch_size, patch_size
    bp = batch_patch.size()[0]
    patch_num = patch_size*patch_size
    batch_size = bp//patch_num
    return batch_patch[:, :, 0, 0]\
            .view([batch_size, patch_num, -1])\
                .permute([0,2,1])\
                    .view([batch_size, -1, patch_size, patch_size])

def collect(hidden_img):
    # batch, dim, patch_size, patch_size -> batch*patch_size*patch_size, dim, 1, 1
    batch, dim, patch_size, _ = hidden_img.size()
    patch_num = patch_size * patch_size
    return hidden_img.view([batch, -1, patch_num])\
        .permute([0,2,1])\
            .view([batch*patch_num, -1, 1, 1])


def img2patch(batch_img: torch.FloatTensor, kernel_size=8):
    # batch, dim, w, h -> batch*(w//patch_size)*(h//patch_size), dim, patch_size, patch_size
    batch, dim, w, h = batch_img.size()
    assert w%kernel_size==0 and h%kernel_size==0
    patch_num = (w//kernel_size) * (h//kernel_size)
    fold = F.unfold(batch_img, kernel_size,1,0,kernel_size)
    return fold.view([batch, dim,kernel_size,kernel_size,patch_num])\
                .permute([0,4,1,2,3])\
                    .reshape([-1, dim, kernel_size, kernel_size])


def patch2img(patches: torch.FloatTensor, img_size=256):
    # batch*(w//patch_size)*(h//patch_size), dim, kernel_size, kernel_size -> batch, dim, w, h
    batch_patch, dim, kernel_size, _ = patches.size()
    patch_size = img_size//kernel_size
    patch_num = patch_size*patch_size
    batch_size = batch_patch // patch_num
    rebuilt = patches.reshape([batch_size, patch_num, -1])\
                .permute([0,2,1])
    return F.fold(rebuilt, img_size, kernel_size, stride=kernel_size)


class LocalVaeModel(torch.nn.Module):

    def __init__(self, hidden_size=16, mid_size=128, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size

        self.enc = Encoder(3, hidden_size,["DownEncoderBlock2D"]*4,[mid_size]*4,double_z=True)

        self.trans = ViTModel(
            ViTConfig(
            hidden_size=mid_size, 
            num_hidden_layers=0,
            num_attention_heads=1,
            intermediate_size=mid_size*8,
            image_size=32,
            patch_size=1,
            num_channels=hidden_size*2,
            encoder_stride=1),
            add_pooling_layer=False)

        self.conv1 = torch.nn.Conv2d(mid_size, hidden_size*2, 1)

        self.trans_back = ViTModel(ViTConfig(
            hidden_size=mid_size, 
            num_hidden_layers=0,
            num_attention_heads=1,
            intermediate_size=mid_size*8,
            image_size=32,
            patch_size=1,
            num_channels=hidden_size,
            encoder_stride=1),
            add_pooling_layer=False)

        self.conv2 = torch.nn.Conv2d(mid_size, hidden_size, 1)

        self.dec = Decoder(hidden_size,3,["UpDecoderBlock2D"]*4, [mid_size]*4)

    def forward(self, batch_img: torch.Tensor) -> torch.Tensor:

        batch_size = batch_img.size()[0]

        # batch, dim, img_size, img_size -> batch * patch_num, dim, patch_size, patch_size
        batch_patch = img2patch(batch_img)

        # batch * patch_num, hidden_size*2, 1, 1
        batch_patch_hidden = self.enc(batch_patch)
        
        # batch, hidden_size*2, patch_size, patch_size
        hidden_img = deploy(batch_patch_hidden)

        # batch, mid_size, patch_size, patch_size
        final_out = self.trans(hidden_img,output_hidden_states=True)['hidden_states'][0][:, 1:, :]\
            .permute([0,2,1])\
                .reshape([batch_size, -1, 32, 32])

        # batch, hidden_size*2, patch_size, patch_size
        final_out = self.conv1(final_out)
   

        dist = DiagonalGaussianDistribution(collect(final_out))

        # batch, hidden_size, 32, 32
        sampled_z = deploy(dist.sample())

        # batch, mid_size, 32, 32
        render_f = self.trans_back(sampled_z,output_hidden_states=True)['hidden_states'][0][:, 1:, :]\
            .permute([0,2,1])\
                .reshape([batch_size, -1, 32, 32])

        # batch, hidden_size, 32, 32
        render_f = self.conv2(render_f)
        
        # batch*patch_num, hidden_size, 1, 1
        render_f = collect(sampled_z)

        # render_f = sampled_z
        rebuilt = self.dec(render_f)\

        pixel = patch2img(rebuilt)

        return pixel, dist.kl()


logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=30000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--unet_mid_size",
        type=int,
        default=128,
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.output_dir is None:
        from datetime import datetime
        args.output_dir = "output_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    return args


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=logging_dir,
    )

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:

        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    vae = LocalVaeModel(hidden_size=args.hidden_size, mid_size=args.unet_mid_size)
    # from utils import TrainMonitor
    # tm = TrainMonitor()
    # tm.register_backward(vae.enc, "enc")
    # tm.register_backward(vae.trans, "trans")
    # tm.register_backward(vae.trans_back, 'transb')
    # tm.register_backward(vae.dec, 'dec')
    
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = vae.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = MsImageDataset(
        MsDataset.load('tany0699/mini_imagenet100', subset_name='default', split='train')
        )
    

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    (
        vae,
        optimizer,
        train_dataloader
    ) = accelerator.prepare(
        vae, optimizer, train_dataloader
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("xavier-test", config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0
    last_save = 0

    for epoch in range(args.num_train_epochs):
        vae.train()

        for step, batch in enumerate(train_dataloader):

            model_pred, kl_loss = vae(batch)

            loss = F.mse_loss(model_pred.float(), batch.float(), reduction="mean") + \
                0.00025 * torch.mean(kl_loss)

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = vae.parameters()
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            optimizer.step()
            progress_bar.update(1)
            optimizer.zero_grad()

            global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            # info_dict = tm.infos
            accelerator.log(logs, step=global_step)

            if step % 100 == 0:
                if accelerator.is_main_process:
                    img = Image.open('/media/xavier/Samsumg/codes/incremental_dm/tmp1.png')
                    x = TRANS(img).unsqueeze(0).to('cuda')
                    vutil.save_image(denormalize(vae(x)[0]), 'tmp.png')

            if global_step >= args.max_train_steps:
                break

            if global_step % args.save_steps == 0:
                if accelerator.is_main_process:
                    torch.save(vae, args.output_dir + "/vae.pt")

    accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:

        print("\n\nLocal-Vae TRAINING DONE!\n\n")
        torch.save(vae, args.output_dir + "/vae.pt")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

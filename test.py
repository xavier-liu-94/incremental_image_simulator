from diffusers.models.autoencoders.vae import Encoder, Decoder, DiagonalGaussianDistribution
import torch
from transformers import ViTModel, ViTConfig
from torch.nn import functional as F
from experiment import LocalVaeModel

m = LocalVaeModel()
print(sum([p.numel() for p in m.parameters()])/(1024*1023))
m.to('cuda')
print(m(torch.randn([8, 3, 256,256], device='cuda')).shape)

# dec = Decoder(8,3,["UpDecoderBlock2D"]*4, [64]*4)
# print(dec(DiagonalGaussianDistribution(torch.randn([16,16,32,32])).sample()))

# m = EncoderModel()
# print(m(torch.randn([16, 3, 256,256])).shape)

# imgs = torch.randn([16, 3, 256,256])
# bimg = F.unfold(imgs, 8,1,0,8)
# print(bimg.shape)
# batch,_,patch = bimg.size()
# n = bimg.view([batch, 3, 8,8,patch])
# print(n[0,:,:,:,0])
# print(imgs[0,:,0:8,0:8])

# enc = Encoder(3, 32,["DownEncoderBlock2D"]*4,[64]*4,double_z=False)

# dummy = torch.randn([100, 3, 8,8])
# out = enc(dummy)
# print(out.shape)

# vit_config = ViTConfig(
# hidden_size=32, 
# num_hidden_layers=4,
# num_attention_heads=4,
# intermediate_size=512,
# image_size=32,
# patch_size=1,
# num_channels=32,
# encoder_stride=1)
# trans = ViTModel(vit_config)

# dummy = torch.randn([100, 32, 32, 32])
# out = trans(dummy)
# print(out['last_hidden_state'].shape)
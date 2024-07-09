from experiment import LocalVaeModel
from PIL import Image
import torch
import torchvision.utils as vutil
from torchvision import transforms

def denormalize(images):
        """
        Denormalize an image array to [0,1].
        """
        return (images / 2 + 0.5).clamp(0, 1)

TRANS = transforms.Compose(
            [transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])]
        )

img = Image.open('/media/xavier/Samsumg/data/pic_1/0e8f09cb9cdf36e421246ee8c0de897e.jpg')
img = img.crop([0,0,256,256])

# m = LocalVaeModel()
# m.load_state_dict(torch.load("./output_example/vae.pt"))
m = torch.load("./output_example2/vae.pt")
m.eval()
m.to('cuda')
x = TRANS(img).unsqueeze(0).to('cuda')
vutil.save_image(denormalize(x), 'tmp1.png')
vutil.save_image(denormalize(m(x)[0]), 'tmp.png')
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from kornia.filters import Canny
from einops import rearrange

canny = Canny().to('cuda')

def test_canny():
    img_path = "/dir/1/1.png"
    image = Image.open(img_path)

    image = np.asarray(image)[:, :, :3]

    image = torch.from_numpy(image).float() / 127. - 1

    image = rearrange(image, '(b h) w c -> b c h w', b=1)
    _, canny_edges = canny(image)

    out = T.ToPILImage()(canny_edges.squeeze(0))

    out.save('./test.png')
    print(canny_edges)

def item_norm(item):
    return item / 127. - 1


# Defaults to the VAE scale of 8. TODO: Change this to be changeable.
def resize_item(item, resize_factor=8):
    f, c, h, w = item.shape
    resize = T.Resize(size=(h // resize_factor, w // resize_factor))

    return resize(item)

def canny_processor(item, img_path=None, from_dataloader=True):
   
    if img_path is not None and not from_dataloader:
        get_img = Image.open(img_path)

        # Remove the alpha channel if the input has one
        np_img = np.asarray(get_img)[:, :, :3]
        image = item_norm(torch.from_numpy(np_img).float())
        image = rearrange(image, '(b h) w c -> b c h w', b=1)
        item = image
    else:
        item = item_norm(item.float())

    _, canny_edges = canny(item)

    return resize_item(canny_edges)

def threshold_processor(item):
    item_processed = resize_item(item_norm(item)) > 0.5

    # The VAE for Stable Diffusion has 4 channels, so we add an arbitrary one (RGB 3 + 1A).
    null_channel = torch.zeros_like(item_processed)[:, :1, ...]
    item_processed = torch.cat([item_processed, null_channel], dim=1)

    return item_processed

ConditionProcessors = dict(
    canny=canny_processor,
    threshold=threshold_processor
)

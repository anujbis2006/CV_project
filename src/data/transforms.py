import torch
from torchvision.transforms import v2
def build_image_transform(image_size, train=False):
    layers=[v2.ToImage(),v2.Resize((image_size,image_size),antialias=True)]
    if train: layers.append(v2.ColorJitter(brightness=.08,contrast=.08))
    return v2.Compose(layers+[v2.ToDtype(torch.float32,scale=True),v2.Normalize((.5,)*3,(.5,)*3)])

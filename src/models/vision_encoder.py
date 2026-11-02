import torch
from torch import nn
class PatchVisionEncoder(nn.Module):
    def __init__(self,h,patch=16): super().__init__(); self.net=nn.Sequential(nn.Conv2d(3,h,patch,patch),nn.Flatten(2)); self.norm=nn.LayerNorm(h)
    def forward(self,x): return self.norm(self.net(x).transpose(1,2))

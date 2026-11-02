from torch import nn
class VLMProjection(nn.Module):
    def __init__(self,input_size,output_size): super().__init__();self.net=nn.Sequential(nn.LayerNorm(input_size),nn.Linear(input_size,output_size))
    def forward(self,x):return self.net(x)

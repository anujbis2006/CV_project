import torch
from torch import nn
class TinyCausalDecoder(nn.Module):
    def __init__(self,vocab,h,layers,heads,dropout):
        super().__init__();self.embedding=nn.Embedding(vocab,h,padding_idx=0);self.decoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(h,heads,h*4,dropout,batch_first=True,norm_first=True,activation='gelu'),layers,enable_nested_tensor=False);self.head=nn.Linear(h,vocab,bias=False)
    def forward(self,prefix,answers):
        inputs=torch.cat([torch.full_like(answers[:,:1],1),answers[:,:-1]],1);seq=torch.cat([prefix,self.embedding(inputs)],1);n=seq.size(1);causal=torch.triu(torch.ones(n,n,device=seq.device,dtype=torch.bool),1)
        return self.head(self.decoder(seq,mask=causal)[:,-answers.size(1):])

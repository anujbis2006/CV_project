"""Paper Sec. 3.1: OCR text/layout + instruction + learned queries in one encoder."""
import torch
from torch import nn
class OCRLayoutEncoder(nn.Module):
    def __init__(self,vocab,h,queries,layers,heads,dropout):
        super().__init__();self.char=nn.Embedding(vocab,h,padding_idx=0);self.box=nn.ModuleList([nn.Embedding(1001,h) for _ in range(4)]);self.queries=nn.Parameter(torch.randn(1,queries,h)*h**-.5);self.nq=queries
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(h,heads,h*4,dropout,batch_first=True,norm_first=True,activation='gelu'),layers,enable_nested_tensor=False)
    def forward(self,ids,boxes,mask,instruction):
        chars=ids.ne(0); words=(self.char(ids)*chars[...,None]).sum(2)/chars.sum(2,keepdim=True).clamp_min(1); layout=sum(e(boxes[...,i].clamp(0,1000)) for i,e in enumerate(self.box)); q=self.queries.expand(ids.size(0),-1,-1); seq=torch.cat([words+layout,self.char(instruction),q],1); padding=torch.cat([~mask,instruction.eq(0),torch.zeros_like(q[...,0],dtype=torch.bool)],1)
        return self.encoder(seq,src_key_padding_mask=padding)[:,-self.nq:]

"""Run actual OCR and the trained local model on one image/question."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from src.data.transforms import build_image_transform
from src.data.vocab import CharTokenizer
from src.models import DocVLM
from src.ocr import TesseractEngine
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
def main():
 p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--question',required=True);p.add_argument('--checkpoint',default='checkpoints/best.pt');p.add_argument('--config',default='configs/debug.yaml');a=p.parse_args();c=load_config(a.config);im=Image.open(a.image).convert('RGB');w,h=im.size;ocr=TesseractEngine().extract(a.image)[:c['max_ocr_tokens']];tok=CharTokenizer(); words=[tok.encode(x['text'],24) for x in ocr]; max_chars=max(map(len,words),default=1); ids=torch.zeros(1,max(1,len(words)),max_chars,dtype=torch.long)
 for index,word in enumerate(words): ids[0,index,:len(word)]=torch.tensor(word)
 sample={'pixel_values':build_image_transform(c['image_size'])(im).unsqueeze(0),'ocr_ids':ids,'ocr_boxes':torch.tensor([[[round(x['box'][0]/w*1000),round(x['box'][1]/h*1000),round(x['box'][2]/w*1000),round(x['box'][3]/h*1000)] for x in ocr]] or [[[0,0,0,0]]]),'ocr_mask':torch.ones(1,max(1,len(ocr)),dtype=torch.bool),'question_ids':torch.tensor([tok.encode(a.question,c['max_question_length'])])}
 # Greedy local decoding. It uses the trained model, but quality is limited by synthetic training.
 model=DocVLM(c);load_checkpoint(model,a.checkpoint);model.eval();answer=torch.tensor([[tok.bos_id]])
 with torch.no_grad():
  for _ in range(c['max_answer_length']):
   out=model(**sample,answer_ids=answer)['logits'][:,-1].argmax(-1,keepdim=True);answer=torch.cat([answer,out],1)
   if out.item()==tok.eos_id:break
 print(json.dumps({'answer':tok.decode(answer[0,1:].tolist()),'ocr_words':ocr},indent=2))
if __name__=='__main__':main()

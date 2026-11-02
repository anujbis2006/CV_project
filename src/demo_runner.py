"""Neutral utilities used by the two clearly separated demonstration runners."""
import time,tracemalloc
from pathlib import Path
import torch
from PIL import Image
from src.data.transforms import build_image_transform
from src.data.vocab import CharTokenizer
from src.ocr import TesseractEngine

def make_input(image_path, question, config):
    image=Image.open(image_path).convert('RGB'); width,height=image.size; words=TesseractEngine().extract(image_path)[:config['max_ocr_tokens']]; tokenizer=CharTokenizer(); encoded=[tokenizer.encode(x['text'],24) for x in words]; chars=max(map(len,encoded),default=1); ids=torch.zeros(1,max(1,len(encoded)),chars,dtype=torch.long)
    for index,word in enumerate(encoded): ids[0,index,:len(word)]=torch.tensor(word)
    boxes=[[round(x['box'][0]/width*1000),round(x['box'][1]/height*1000),round(x['box'][2]/width*1000),round(x['box'][3]/height*1000)] for x in words] or [[0,0,0,0]]
    return {'pixel_values':build_image_transform(config['image_size'])(image).unsqueeze(0),'ocr_ids':ids,'ocr_boxes':torch.tensor([boxes]),'ocr_mask':torch.ones(1,max(1,len(words)),dtype=torch.bool),'question_ids':torch.tensor([tokenizer.encode(question,config['max_question_length'])])},words,tokenizer

def execute(model, inputs, tokenizer, max_tokens):
    model.eval(); answer=torch.tensor([[tokenizer.bos_id]]); tracemalloc.start(); start=time.perf_counter()
    with torch.no_grad():
        for _ in range(max_tokens):
            result=model(**inputs,answer_ids=answer); next_token=result['logits'][:,-1].argmax(-1,keepdim=True); answer=torch.cat([answer,next_token],1)
            if next_token.item()==tokenizer.eos_id: break
    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    return {'answer':tokenizer.decode(answer[0,1:].tolist()),'latency_ms':(time.perf_counter()-start)*1000,'memory_mb':peak/(1024**2),'output_tokens':answer.size(1)-1,'result':result}

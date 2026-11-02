"""DocVQA-style reader: images plus OCR words/boxes, question and answer."""
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from src.data.transforms import build_image_transform
from src.data.vocab import CharTokenizer
class DocumentVQADataset(Dataset):
    def __init__(self,root_dir,split,image_size,max_ocr_tokens,max_question_length,max_answer_length,train=False):
        self.root=Path(root_dir); manifest=self.root/'processed'/f'{split}.jsonl'
        if not manifest.exists(): raise FileNotFoundError(f'Missing {manifest}; run scripts/generate_synthetic_data.py.')
        self.records=[json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]; self.transform=build_image_transform(image_size,train); self.tok=CharTokenizer()
        self.max_ocr_tokens,self.max_question_length,self.max_answer_length=max_ocr_tokens,max_question_length,max_answer_length
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]; answer=r['answers'][0] if isinstance(r['answers'],list) else r['answers']; words=r.get('ocr_tokens',[])[:self.max_ocr_tokens]
        return {'pixel_values':self.transform(Image.open(self.root/r['image']).convert('RGB')),'ocr_ids':[self.tok.encode(w,24) for w in words], 'ocr_boxes':r.get('ocr_boxes',[])[:self.max_ocr_tokens], 'question_ids':self.tok.encode(r['question'],self.max_question_length), 'answer_ids':self.tok.encode(answer,self.max_answer_length), 'answer_text':answer,'id':r.get('id',str(i))}

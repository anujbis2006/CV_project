import torch
class DocumentCollator:
    def __call__(self,batch):
        words=max(1,max(len(x['ocr_ids']) for x in batch)); chars=max(1,max((len(w) for x in batch for w in x['ocr_ids']),default=1)); ocr=torch.zeros(len(batch),words,chars,dtype=torch.long); boxes=torch.zeros(len(batch),words,4,dtype=torch.long); mask=torch.zeros(len(batch),words,dtype=torch.bool)
        for i,x in enumerate(batch):
            for j,w in enumerate(x['ocr_ids']): ocr[i,j,:len(w)]=torch.tensor(w); mask[i,j]=True
            if x['ocr_boxes']: boxes[i,:len(x['ocr_boxes'])]=torch.tensor(x['ocr_boxes'])
        def pad(key):
            n=max(len(x[key]) for x in batch); out=torch.zeros(len(batch),n,dtype=torch.long)
            for i,x in enumerate(batch):out[i,:len(x[key])]=torch.tensor(x[key])
            return out
        answers=pad('answer_ids'); labels=answers.masked_fill(answers.eq(0),-100)
        return {'pixel_values':torch.stack([x['pixel_values'] for x in batch]),'ocr_ids':ocr,'ocr_boxes':boxes,'ocr_mask':mask,'question_ids':pad('question_ids'),'answer_ids':answers,'labels':labels,'answer_text':[x['answer_text'] for x in batch],'id':[x['id'] for x in batch]}

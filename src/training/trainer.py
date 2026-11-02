from pathlib import Path
import torch
from src.losses import autoregressive_cross_entropy
from src.utils.checkpoint import save_checkpoint
class Trainer:
    def __init__(self, model, optimizer, scheduler, device, config): self.model,self.optimizer,self.scheduler,self.device,self.config=model,optimizer,scheduler,device,config
    def _run(self, loader, training):
        self.model.train(training); total=0.
        for batch in loader:
            batch={key:value.to(self.device) if isinstance(value,torch.Tensor) else value for key,value in batch.items()}; keys=('pixel_values','ocr_ids','ocr_boxes','ocr_mask','question_ids','answer_ids')
            with torch.set_grad_enabled(training):
                loss=autoregressive_cross_entropy(self.model(**{key:batch[key] for key in keys})['logits'],batch['labels'])
                if training: self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.); self.optimizer.step(); self.scheduler.step()
            total += loss.item()
        return total/max(1,len(loader))
    def fit(self, train, val):
        best=float('inf'); history=[]
        for epoch in range(1,self.config['epochs']+1):
            row={'epoch':epoch,'train_loss':self._run(train,True),'val_loss':self._run(val,False)}; history.append(row)
            if row['val_loss']<best: best=row['val_loss']; save_checkpoint(self.model,self.optimizer,self.scheduler,str(Path(self.config['checkpoint_dir'])/'best.pt'),epoch,row)
        return history

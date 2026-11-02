from torch.utils.data import DataLoader
from src.data.collator import DocumentCollator
from src.data.dataset import DocumentVQADataset
def build_loader(data_dir,split,config,shuffle=None):
    d=DocumentVQADataset(data_dir,split,config['image_size'],config['max_ocr_tokens'],config['max_question_length'],config['max_answer_length'],split=='train')
    return DataLoader(d,batch_size=config['batch_size'],shuffle=split=='train' if shuffle is None else shuffle,num_workers=config.get('num_workers',0),collate_fn=DocumentCollator())

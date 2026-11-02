import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataloader import build_loader
from src.models import DocVLM
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.device import get_torch_device
from src.utils.seed import set_seed
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='configs/debug.yaml'); args=parser.parse_args(); config=load_config(args.config); set_seed(config['seed']); device=get_torch_device(config['device']); model=DocVLM(config).to(device)
    train=build_loader(config['data_dir'],'train',config); val=build_loader(config['data_dir'],'val',config,False); optimizer=build_optimizer(model,config); scheduler=build_scheduler(optimizer,config['epochs']*len(train),config['warmup_steps']); print(Trainer(model,optimizer,scheduler,device,config).fit(train,val))
if __name__=='__main__': main()

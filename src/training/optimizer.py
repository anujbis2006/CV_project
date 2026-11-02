import torch

def build_optimizer(model, config):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'], weight_decay=config.get('weight_decay', .01))

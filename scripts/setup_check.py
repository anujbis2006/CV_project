import platform
import torch, torchvision, PIL, yaml
print('Python:',platform.python_version()); print('torch:',torch.__version__); print('torchvision:',torchvision.__version__); print('Pillow:',PIL.__version__); print('PyYAML:',yaml.__version__); print('CUDA:',torch.cuda.is_available()); print('MPS:',hasattr(torch.backends,'mps') and torch.backends.mps.is_available()); print('Selected:', 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else 'cpu')

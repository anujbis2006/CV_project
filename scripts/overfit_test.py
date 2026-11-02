"""A short optimisation check. It uses synthetic records and does not report paper performance."""
import subprocess,sys
subprocess.run([sys.executable,'scripts/generate_synthetic_data.py','--samples','12'],check=True)
subprocess.run([sys.executable,'scripts/train.py','--config','configs/debug.yaml'],check=True)
print('Overfit diagnostic completed; inspect decreasing train_loss above.')

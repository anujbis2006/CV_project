import subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,'scripts/generate_synthetic_data.py','--samples','12'],cwd=root,check=True)
subprocess.run([sys.executable,'scripts/train.py','--config','configs/debug.yaml'],cwd=root,check=True)
assert (root/'checkpoints'/'best.pt').exists()
print('SMOKE TEST PASSED: data -> loader -> forward/backward -> optimizer -> checkpoint')

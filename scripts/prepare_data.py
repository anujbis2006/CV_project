"""Validate externally prepared JSONL manifests. OCR must be supplied by the user/dataset."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',default='data/processed/train.jsonl');a=p.parse_args(); lines=Path(a.input).read_text().splitlines(); [json.loads(x) for x in lines if x.strip()]; Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text('\n'.join(lines)+'\n');print('Validated manifest:',a.output)

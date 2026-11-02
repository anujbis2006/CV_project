"""Measure actual local forward latency/token counts for fixed or adaptive query counts."""
import argparse,csv,json,sys,time,tracemalloc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.data.dataloader import build_loader
from src.models import DocVLM
from src.utils.config import load_config
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/debug.yaml');p.add_argument('--queries',default='8,16,32');a=p.parse_args();base=load_config(a.config);out=Path('outputs/benchmarks/query_efficiency');out.mkdir(parents=True,exist_ok=True);records=[]
 for k in map(int,a.queries.split(',')):
  c={**base,'num_queries':k,'adaptive_k':min(k,base.get('adaptive_k',k))}; model=DocVLM(c).eval();batch=next(iter(build_loader(c['data_dir'],'test',c,False))); inputs={x:batch[x] for x in ('pixel_values','ocr_ids','ocr_boxes','ocr_mask','question_ids','answer_ids')}
  with torch.no_grad():model(**inputs)
  tracemalloc.start();start=time.perf_counter();
  with torch.no_grad(): result=model(**inputs)
  latency=(time.perf_counter()-start)*1000;_,peak=tracemalloc.get_traced_memory();tracemalloc.stop();records.append({'candidate_queries':k,'selected_queries':result['ocr_features'].shape[1],'visual_tokens':result['visual_features'].shape[1],'total_prefix_tokens':result['ocr_features'].shape[1]+result['visual_features'].shape[1],'latency_ms':round(latency,3),'tracemalloc_peak_bytes':peak})
 with (out/'query_efficiency.csv').open('w',newline='') as f: writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
 (out/'query_efficiency.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
if __name__=='__main__':main()

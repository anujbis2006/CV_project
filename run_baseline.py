import argparse
from baseline import BaselineDocVLM
from src.demo_runner import execute,make_input
from src.utils.config import load_config
def main():
 p=argparse.ArgumentParser();p.add_argument('--image',default='data/synthetic/0000.png');p.add_argument('--question',default='What is the total amount?');p.add_argument('--config',default='configs/baseline.yaml');a=p.parse_args();c=load_config(a.config);inputs,words,tok=make_input(a.image,a.question,c);model=BaselineDocVLM(c);out=execute(model,inputs,tok,c['max_answer_length']);visual=out['result']['visual_features'].shape[1]
 print('\n========================================\n[BASELINE / PAPER IMPLEMENTATION]\n========================================');print('Model: local fixed-query DocVLM structural baseline');print(f'Query mode: Fixed\nNumber of queries: {c["num_queries"]}\nInput prefix tokens: {visual+c["num_queries"]}\nOutput tokens: {out["output_tokens"]}\nLatency: {out["latency_ms"]:.2f} ms\nPython peak memory: {out["memory_mb"]:.3f} MB\nConfig: {a.config}\nQuestion: {a.question}\nAnswer: {out["answer"]!r}')
if __name__=='__main__':main()

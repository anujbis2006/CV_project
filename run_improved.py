import argparse
from improved import ImprovedDocVLM
from src.demo_runner import execute,make_input
from src.utils.config import load_config
def main():
 p=argparse.ArgumentParser();p.add_argument('--image',default='data/synthetic/0000.png');p.add_argument('--question',default='What is the total amount?');p.add_argument('--config',default='configs/improved.yaml');a=p.parse_args();c=load_config(a.config);inputs,words,tok=make_input(a.image,a.question,c);model=ImprovedDocVLM(c);out=execute(model,inputs,tok,c['max_answer_length']);result=out['result']; selected=result['ocr_features'].shape[1];indices=result['selected_query_indices'][0].tolist();visual=result['visual_features'].shape[1]
 print('\n========================================\n[IMPROVED / OUR PROPOSED EXTENSION]\n========================================');print('Model: local DocVLM + adaptive question-aware query selection');print(f'Query mode: Adaptive + Question-Aware\nCandidate/maximum queries: {c["max_queries"]}\nSelected queries: {selected}\nSelected indices: {indices}\nInput prefix tokens: {visual+selected}\nOutput tokens: {out["output_tokens"]}\nLatency: {out["latency_ms"]:.2f} ms\nPython peak memory: {out["memory_mb"]:.3f} MB\nConfig: {a.config}\nQuestion: {a.question}\nAnswer: {out["answer"]!r}')
if __name__=='__main__':main()

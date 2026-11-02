import argparse
from baseline import BaselineDocVLM
from improved import ImprovedDocVLM
from src.demo_runner import execute,make_input
from src.utils.config import load_config
def main():
 p=argparse.ArgumentParser();p.add_argument('--image',default='data/synthetic/0000.png');p.add_argument('--question',default='What is the total amount?');a=p.parse_args();bc,ic=load_config('configs/baseline.yaml'),load_config('configs/improved.yaml');bi,_,tok=make_input(a.image,a.question,bc);ii,_,_=make_input(a.image,a.question,ic);base=execute(BaselineDocVLM(bc),bi,tok,bc['max_answer_length']);improved=execute(ImprovedDocVLM(ic),ii,tok,ic['max_answer_length']);br,ir=base['result'],improved['result'];rows=[('Queries',bc['num_queries'],ir['ocr_features'].shape[1]),('Tokens',br['visual_features'].shape[1]+bc['num_queries'],ir['visual_features'].shape[1]+ir['ocr_features'].shape[1]),('Latency',f'{base["latency_ms"]:.2f} ms',f'{improved["latency_ms"]:.2f} ms'),('Memory',f'{base["memory_mb"]:.3f} MB',f'{improved["memory_mb"]:.3f} MB'),('Answer',repr(base['answer']),repr(improved['answer']))]
 print('\n==================================================\nBASELINE VS IMPROVED\n==================================================\nMetric                 Baseline                 Improved\n--------------------------------------------------');[print(f'{name:<22} {str(left):<24} {right}') for name,left,right in rows]
if __name__=='__main__':main()

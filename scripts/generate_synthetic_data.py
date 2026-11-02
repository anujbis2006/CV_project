"""Create explicitly SYNTHETIC document VQA data; it is not a benchmark substitute."""
import argparse, json, random
from pathlib import Path
from PIL import Image, ImageDraw

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', default='data'); parser.add_argument('--samples', type=int, default=24); args = parser.parse_args()
    root = Path(args.output); (root / 'synthetic').mkdir(parents=True, exist_ok=True); random.seed(42); records = []
    for index in range(args.samples):
        value = str(random.randint(10, 99)); image = Image.new('RGB', (480, 240), 'white'); draw = ImageDraw.Draw(image)
        draw.text((35, 45), f'ITEM {index}', fill='black'); draw.text((35, 125), f'TOTAL: {value}', fill='black'); filename = f'{index:04}.png'; image.save(root / 'synthetic' / filename)
        records.append({'id':str(index), 'image':f'synthetic/{filename}', 'question':'What is the total?', 'answers':[value], 'ocr_tokens':[f'ITEM {index}', 'TOTAL', value], 'ocr_boxes':[[70,180,350,260],[70,500,260,620],[300,500,430,620]], 'synthetic':True})
    cuts = (max(2, int(.7 * len(records))), max(3, int(.85 * len(records)))); out = root / 'processed'; out.mkdir(exist_ok=True)
    for split, rows in [('train', records[:cuts[0]]), ('val', records[cuts[0]:cuts[1]]), ('test', records[cuts[1]:])]: (out / f'{split}.jsonl').write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    print(f'Created SYNTHETIC data in {root}.')
if __name__ == '__main__': main()

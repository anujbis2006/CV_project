import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from src.ocr import TesseractEngine
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args(); width,height=Image.open(a.input).size; rows=TesseractEngine().extract(a.input)
 for row in rows: row['normalized_box']=[round(row['box'][0]/width*1000),round(row['box'][1]/height*1000),round(row['box'][2]/width*1000),round(row['box'][3]/height*1000)]
 Path(a.output).write_text(json.dumps({'image':a.input,'words':rows},indent=2));print(f'Wrote {len(rows)} OCR words to {a.output}')
if __name__=='__main__':main()

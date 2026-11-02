import csv,subprocess
from pathlib import Path
from src.ocr.base import OCREngine
class TesseractEngine(OCREngine):
    def extract(self,image_path):
        result=subprocess.run(['tesseract',str(image_path),'stdout','tsv'],check=True,capture_output=True,text=True); rows=[]
        for row in csv.DictReader(result.stdout.splitlines(),delimiter='\t'):
            text=(row.get('text') or '').strip()
            if not text: continue
            x,y,w,h=(int(row[key]) for key in ('left','top','width','height')); rows.append({'text':text,'box':[x,y,x+w,y+h],'confidence':float(row.get('conf','-1'))})
        return rows

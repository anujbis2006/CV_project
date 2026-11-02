"""Paper datasets have separate licenses; prints safe manual acquisition guidance."""
print('No dataset is downloaded automatically. Obtain DocVQA/InfoVQA/etc. under their licenses, run OCR, then use scripts/prepare_data.py with JSONL records containing image, question, answers, ocr_tokens, and normalized ocr_boxes [0,1000].')

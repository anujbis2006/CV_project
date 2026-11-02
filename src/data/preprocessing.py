from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image


@dataclass
class DocumentSample:
    image: Image.Image
    text: str
    metadata: Dict[str, str]


class SimplePreprocessor:
    def __init__(self, image_size: int = 224, max_text_length: int = 128):
        self.image_size = image_size
        self.max_text_length = max_text_length

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        return image

    def tokenize_text(self, text: str) -> List[int]:
        tokens = text.strip().split()
        if not tokens:
            return [0] * min(8, self.max_text_length)
        out = []
        for token in tokens[: self.max_text_length]:
            out.append(len(token) % 256 + 1)
        if len(out) < self.max_text_length:
            out.extend([0] * (self.max_text_length - len(out)))
        return out[: self.max_text_length]

    def preprocess_sample(self, sample: DocumentSample) -> Dict[str, object]:
        image = self.preprocess_image(sample.image)
        token_ids = self.tokenize_text(sample.text)
        return {
            "image": np.asarray(image, dtype=np.float32) / 255.0,
            "text_tokens": np.asarray(token_ids, dtype=np.int64),
            "text": sample.text,
            "metadata": sample.metadata,
        }


class SyntheticDocumentDataset:
    def __init__(self, num_samples: int = 100, image_size: int = 224, max_text_length: int = 128):
        self.num_samples = num_samples
        self.preprocessor = SimplePreprocessor(image_size=image_size, max_text_length=max_text_length)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, object]:
        image = Image.fromarray(np.random.randint(0, 255, (image_size := self.preprocessor.image_size, image_size, 3), dtype=np.uint8))
        text = f"sample {idx} document text with a few words and numbers 123"
        return self.preprocessor.preprocess_sample(DocumentSample(image=image, text=text, metadata={"id": str(idx)}))

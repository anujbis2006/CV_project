from abc import ABC,abstractmethod
class OCREngine(ABC):
    @abstractmethod
    def extract(self,image_path): """Return [{text, box:[x0,y0,x1,y1], confidence}]."""

"""Fixed-query baseline wrapper. It does not import adaptive-selection code directly."""
from src.models.model import DocVLM
class BaselineDocVLM(DocVLM):
    def __init__(self, config):
        fixed={**config,'compression_type':'fixed','adaptive_queries':False,'question_aware_selection':False}
        super().__init__(fixed)

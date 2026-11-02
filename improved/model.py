"""Our proposed extension wrapper: adaptive question-aware selection only."""
from src.models.model import DocVLM
class ImprovedDocVLM(DocVLM):
    def __init__(self, config):
        improved={**config,'compression_type':'adaptive','adaptive_queries':True,'question_aware_selection':True,'adaptive_k':config.get('adaptive_k',config.get('max_queries',config['num_queries']))}
        super().__init__(improved)

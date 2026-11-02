from baseline import BaselineDocVLM
from improved import ImprovedDocVLM
from src.utils.config import load_config

def test_baseline_is_fixed_and_improved_is_adaptive():
    baseline=BaselineDocVLM(load_config('configs/baseline.yaml'))
    improved=ImprovedDocVLM(load_config('configs/improved.yaml'))
    assert baseline.selector is None
    assert improved.selector is not None
    assert baseline.ocr_encoder.nq == 64

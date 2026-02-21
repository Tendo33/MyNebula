from scripts.evals.run_all_quality_checks import THRESHOLDS


def test_quality_gate_threshold_defaults():
    assert THRESHOLDS["p_at_5"] >= 0.80
    assert THRESHOLDS["coverage"] >= 0.95
    assert THRESHOLDS["cluster_purity"] >= 0.75

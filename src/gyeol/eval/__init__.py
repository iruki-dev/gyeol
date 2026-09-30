"""Evaluation: probes and leakage (M3); reconstruction, vocoder benchmark and re-encoding consistency (M4);
knob recovery (M6); robustness grid, expert benchmarks and coach agreement, listening-test calibration and
model cards (M8)."""

from .benchmarks import (
    LABELS,
    AgreementReport,
    BenchmarkClip,
    BenchmarkFieldMap,
    BenchmarkReport,
    ExpertSegment,
    Prediction,
    coach_agreement,
    cohen_kappa,
    fleiss_kappa,
    load_expert_benchmark,
    predictions_from_explanation,
    score_benchmark,
)
from .listening import AudibilityCalibration, calibrate_audibility
from .model_card import REQUIRED_OUT_OF_SCOPE, EvalTable, ModelCard, card_from_checkpoint
from .probes import LeakageResult, ProbeResult, leakage, probe_battery, probe_classify, probe_regress
from .reconstruction import BenchmarkItem, CategoryReport, log_spectral_distance, reencoding_consistency, vocoder_benchmark
from .robustness import GridCondition, GridItem, RobustnessReport, robustness_grid, run_robustness

__all__ = [
    "LABELS", "AgreementReport", "AudibilityCalibration", "BenchmarkClip", "BenchmarkFieldMap", "BenchmarkItem", "BenchmarkReport",
    "CategoryReport", "EvalTable", "ExpertSegment", "GridCondition", "GridItem", "LeakageResult", "ModelCard", "Prediction",
    "ProbeResult", "REQUIRED_OUT_OF_SCOPE", "RobustnessReport", "calibrate_audibility", "card_from_checkpoint",
    "coach_agreement", "cohen_kappa", "fleiss_kappa", "leakage", "load_expert_benchmark", "log_spectral_distance",
    "predictions_from_explanation", "probe_battery", "probe_classify", "probe_regress", "reencoding_consistency",
    "robustness_grid", "run_robustness", "score_benchmark", "vocoder_benchmark",
]

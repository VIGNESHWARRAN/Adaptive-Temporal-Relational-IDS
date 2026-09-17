"""Evaluation, training, and benchmarking utilities."""
from .leakage_checker import run_mandatory_leakage_checks
from .trainer import ExperimentTrainer

try:
    from .benchmark_harness import BenchmarkEvaluator
    __all__ = [
        "run_mandatory_leakage_checks",
        "ExperimentTrainer",
        "BenchmarkEvaluator",
    ]
except ImportError:
    __all__ = [
        "run_mandatory_leakage_checks",
        "ExperimentTrainer",
    ]

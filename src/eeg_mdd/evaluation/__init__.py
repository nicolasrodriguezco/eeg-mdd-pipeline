"""Evaluación independiente del sujeto: CV anidada, métricas por sujeto, bootstrap y permutaciones."""

from .cv import assert_disjoint, outer_splits
from .metrics import METRICS, aggregate_to_subject, bootstrap_ci, subject_metrics
from .nested import CVResult, nested_cv, permutation_test

__all__ = ["assert_disjoint", "outer_splits", "METRICS", "aggregate_to_subject", "bootstrap_ci",
           "subject_metrics", "CVResult", "nested_cv", "permutation_test"]

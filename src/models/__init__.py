"""Model architectures, encoders, fusion strategies, quantum PQC layers, and loss functions."""

from .encoders import FixedTemporalEncoder, FixedRelationalEncoder
from .fusion import (
    ConcatenationFusion,
    GatedAdaptiveFusion,
    BilinearTensorFusion,
    CrossAttentionFusion,
    DecisionLevelEnsembleFusion,
    QuantumCircuitFusion,
    get_fusion_module,
)
from .classifiers import MultimodalFusionClassifier
from .single_modality import TemporalOnlyClassifier, SpatialRelationalOnlyClassifier
from .losses import (
    StandardCrossEntropyLoss,
    ClassWeightedCrossEntropyLoss,
    FocalLoss,
    GradientAdaptiveHybridLoss,
)

__all__ = [
    "FixedTemporalEncoder",
    "FixedRelationalEncoder",
    "ConcatenationFusion",
    "GatedAdaptiveFusion",
    "BilinearTensorFusion",
    "CrossAttentionFusion",
    "DecisionLevelEnsembleFusion",
    "QuantumCircuitFusion",
    "get_fusion_module",
    "MultimodalFusionClassifier",
    "TemporalOnlyClassifier",
    "SpatialRelationalOnlyClassifier",
    "StandardCrossEntropyLoss",
    "ClassWeightedCrossEntropyLoss",
    "FocalLoss",
    "GradientAdaptiveHybridLoss",
]

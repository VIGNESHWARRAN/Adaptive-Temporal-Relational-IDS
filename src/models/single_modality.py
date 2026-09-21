"""Single-Modality Classifiers for Modality Ablation Experiments.

Contains:
1. TemporalOnlyClassifier: Uses ONLY FixedTemporalEncoder (GRU).
2. SpatialRelationalOnlyClassifier: Uses ONLY FixedRelationalEncoder (Graph Relational).
"""

from typing import List, Dict, Any, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import FixedTemporalEncoder, FixedRelationalEncoder


class TemporalOnlyClassifier(nn.Module):
    """Classifier using ONLY the Temporal Sequence Encoder (FixedTemporalEncoder).

    Input: Temporal sequence feature tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim).
    No relational or graph features are provided or processed.
    """

    def __init__(
        self,
        temp_input_dim: int = 13,
        embed_dim: int = 64,
        num_classes: int = 2,
        dropout_rate: float = 0.2,
        seq_input_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        t_in = seq_input_dim if seq_input_dim is not None else temp_input_dim
        self.num_classes = num_classes
        self.temp_encoder = FixedTemporalEncoder(
            input_dim=t_in, hidden_dim=embed_dim, embed_dim=embed_dim, num_layers=2
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_rel: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass using ONLY temporal sequence features."""
        h_temp = self.temp_encoder(x_seq)
        logits = self.classifier_head(h_temp)
        return logits

    def get_reference_parameters(self) -> List[nn.Parameter]:
        """Returns classifier head parameters as reference parameter set for loss gradient norm calculation."""
        return list(self.classifier_head.parameters())

    def get_parameter_count(self) -> Dict[str, int]:
        """Returns total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "encoder_parameters": sum(p.numel() for p in self.temp_encoder.parameters() if p.requires_grad),
            "head_parameters": sum(p.numel() for p in self.classifier_head.parameters() if p.requires_grad),
        }


class SpatialRelationalOnlyClassifier(nn.Module):
    """Classifier using ONLY the Relational Graph Encoder (FixedRelationalEncoder).

    Input: Relational node feature tensor of shape (batch_size, node_input_dim) and optional edge tensor.
    No temporal sequence features are provided or processed.
    """

    def __init__(
        self,
        rel_node_dim: int = 5,
        rel_edge_dim: int = 0,
        embed_dim: int = 64,
        num_classes: int = 2,
        dropout_rate: float = 0.2,
        node_input_dim: Optional[int] = None,
        edge_input_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        n_in = node_input_dim if node_input_dim is not None else rel_node_dim
        e_in = edge_input_dim if edge_input_dim is not None else rel_edge_dim
        self.num_classes = num_classes
        self.rel_encoder = FixedRelationalEncoder(
            node_input_dim=n_in, edge_input_dim=e_in, embed_dim=embed_dim
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes),
        )

    def forward(
        self,
        x_seq: Optional[torch.Tensor] = None,
        x_rel: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass using ONLY relational graph features."""
        rel_inputs = x_rel if x_rel is not None else x_seq
        h_rel = self.rel_encoder(rel_inputs)
        logits = self.classifier_head(h_rel)
        return logits

    def get_reference_parameters(self) -> List[nn.Parameter]:
        """Returns classifier head parameters as reference parameter set for loss gradient norm calculation."""
        return list(self.classifier_head.parameters())

    def get_parameter_count(self) -> Dict[str, int]:
        """Returns total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "encoder_parameters": sum(p.numel() for p in self.rel_encoder.parameters() if p.requires_grad),
            "head_parameters": sum(p.numel() for p in self.classifier_head.parameters() if p.requires_grad),
        }

"""Multimodal Adaptive Temporal-Relational Classifier.

Integrates fixed temporal and relational encoders with configurable fusion modules and classifier head.
"""

from typing import List, Dict, Any, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import FixedTemporalEncoder, FixedRelationalEncoder
from .fusion import (
    ConcatenationFusion,
    DecisionLevelEnsembleFusion,
    get_fusion_module,
)


class MultimodalFusionClassifier(nn.Module):
    """Multimodal Fusion Classifier for Adaptive Temporal-Relational Intrusion Detection.

    Supports:
    - 6 Controlled Fusion Modules (Concatenation, Gated, Bilinear, Cross-Attention, Decision Ensemble, Quantum PQC).
    - Dual input API: 3D sequence tensors + graph snapshots OR batched node/edge feature tensors.
    """

    def __init__(
        self,
        fusion_type: str = "concat",
        seq_input_dim: int = 13,
        node_input_dim: int = 5,
        edge_input_dim: int = 13,
        embed_dim: int = 64,
        num_classes: int = 2,
        dropout_rate: float = 0.2,
        # Alias parameters for Phase 1 trainer compatibility
        temp_input_dim: Optional[int] = None,
        rel_node_dim: Optional[int] = None,
        rel_edge_dim: Optional[int] = None,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.num_classes = num_classes

        t_in = temp_input_dim if temp_input_dim is not None else seq_input_dim
        n_in = rel_node_dim if rel_node_dim is not None else node_input_dim
        e_in = rel_edge_dim if rel_edge_dim is not None else edge_input_dim

        self.temp_encoder = FixedTemporalEncoder(
            input_dim=t_in, hidden_dim=embed_dim, embed_dim=embed_dim, num_layers=2
        )
        self.rel_encoder = FixedRelationalEncoder(
            node_input_dim=n_in, edge_input_dim=e_in, embed_dim=embed_dim
        )
        self.fusion_module = get_fusion_module(
            fusion_type, embed_dim=embed_dim, num_classes=num_classes
        )

        if not isinstance(self.fusion_module, DecisionLevelEnsembleFusion):
            self.classifier_head = nn.Sequential(
                nn.Linear(embed_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(32, num_classes),
            )
        else:
            self.classifier_head = None

    def forward(
        self,
        x_seq: Union[torch.Tensor, Any],
        batch_graphs: Optional[Union[List[Dict[str, torch.Tensor]], torch.Tensor, Any]] = None,
        rel_edge_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_seq: Temporal sequence tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim).
            batch_graphs: Graph snapshot objects OR node feature tensor of shape (batch_size, node_dim).
            rel_edge_x: Optional relational edge feature tensor.

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # If batch_graphs is None and rel_edge_x is passed, treat x_seq as temp_x and batch_graphs as rel_node_x
        h_temp = self.temp_encoder(x_seq)

        if batch_graphs is not None:
            if isinstance(batch_graphs, torch.Tensor) and rel_edge_x is not None:
                h_rel = self.rel_encoder(batch_graphs, rel_edge_x)
            else:
                h_rel = self.rel_encoder(batch_graphs)
        else:
            h_rel = self.rel_encoder(x_seq)

        fused_out = self.fusion_module(h_temp, h_rel)

        if self.classifier_head is not None:
            logits = self.classifier_head(fused_out)
        else:
            logits = fused_out

        return logits

    def get_reference_parameters(self) -> List[nn.Parameter]:
        """Returns reference parameters used for gradient norm calculation in adaptive loss functions."""
        if self.classifier_head is not None:
            return list(self.classifier_head.parameters())
        else:
            return list(self.fusion_module.parameters())


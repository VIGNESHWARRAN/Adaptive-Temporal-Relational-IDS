"""Unified Multimodal Fusion Classifier for Phase 1 experimentation.

Integrates FixedTemporalEncoder, FixedRelationalEncoder, ConcatenationFusion,
and the fixed Classifier Head.
"""

import torch
import torch.nn as nn
from .encoders import FixedTemporalEncoder, FixedRelationalEncoder
from .fusion import ConcatenationFusion


class MultimodalFusionClassifier(nn.Module):
    """Multimodal Fusion Classifier for Adaptive Temporal-Relational Intrusion Detection.
    
    Architecture:
    - FixedTemporalEncoder (hidden_dim=64)
    - FixedRelationalEncoder (hidden_dim=64)
    - ConcatenationFusion (fused_dim=64)
    - Classifier Head: Linear(64 -> 32) -> ReLU -> Dropout(0.2) -> Linear(32 -> num_classes)
    """

    def __init__(
        self,
        temp_input_dim: int,
        rel_node_dim: int,
        rel_edge_dim: int = 0,
        num_classes: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.temporal_encoder = FixedTemporalEncoder(input_dim=temp_input_dim, hidden_dim=64, num_layers=2)
        self.relational_encoder = FixedRelationalEncoder(node_dim=rel_node_dim, edge_dim=rel_edge_dim, hidden_dim=64)
        self.fusion = ConcatenationFusion(temp_dim=64, rel_dim=64, fused_dim=64)

        self.classifier_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes),
        )

    def forward(
        self,
        temp_x: torch.Tensor,
        rel_node_x: torch.Tensor,
        rel_edge_x: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            temp_x: Temporal sequence features (batch_size, seq_len, temp_input_dim)
            rel_node_x: Relational node features (batch_size, rel_node_dim)
            rel_edge_x: Optional relational edge features (batch_size, rel_edge_dim)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        temp_emb = self.temporal_encoder(temp_x)
        rel_emb = self.relational_encoder(rel_node_x, rel_edge_x)
        fused_emb = self.fusion(temp_emb, rel_emb)
        logits = self.classifier_head(fused_emb)
        return logits

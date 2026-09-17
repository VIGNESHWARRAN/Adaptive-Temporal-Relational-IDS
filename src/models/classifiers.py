"""
Multimodal Adaptive Temporal-Relational Classifier.
Integrates fixed temporal and relational encoders with any of the 6 fusion modules under test.
"""

from typing import List, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import FixedTemporalEncoder, FixedRelationalEncoder
from .fusion import get_fusion_module, DecisionLevelEnsembleFusion


class MultimodalFusionClassifier(nn.Module):
    def __init__(
        self,
        fusion_type: str = "concat",
        seq_input_dim: int = 13,
        node_input_dim: int = 5,
        edge_input_dim: int = 13,
        embed_dim: int = 64,
        num_classes: int = 2,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.temp_encoder = FixedTemporalEncoder(input_dim=seq_input_dim, embed_dim=embed_dim)
        self.rel_encoder = FixedRelationalEncoder(node_input_dim=node_input_dim, edge_input_dim=edge_input_dim, embed_dim=embed_dim)
        self.fusion_module = get_fusion_module(fusion_type, embed_dim=embed_dim, num_classes=num_classes)

        if not isinstance(self.fusion_module, DecisionLevelEnsembleFusion):
            self.classifier_head = nn.Sequential(
                nn.Linear(embed_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, num_classes),
            )
        else:
            self.classifier_head = None

    def forward(
        self, x_seq: torch.Tensor, batch_graphs: Union[List[Dict[str, torch.Tensor]], Any]
    ) -> torch.Tensor:
        # Extract fixed representations
        h_temp = self.temp_encoder(x_seq)
        h_rel = self.rel_encoder(batch_graphs)

        # Fuse representations using selected fusion strategy (1 of 6)
        fused_out = self.fusion_module(h_temp, h_rel)

        if self.classifier_head is not None:
            logits = self.classifier_head(fused_out)
        else:
            # DecisionLevelEnsembleFusion returns logits directly
            logits = fused_out

        return logits

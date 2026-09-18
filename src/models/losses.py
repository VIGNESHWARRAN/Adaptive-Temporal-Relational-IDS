"""Loss function definitions for Phase 1 Controlled Experiments.

Contains:
1. StandardCrossEntropyLoss (Unweighted Cross Entropy Baseline)
2. ClassWeightedCrossEntropyLoss (Inverse Class Frequency Weighted Cross Entropy)
3. FocalLoss (Multiclass Focal Loss with gamma=2.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardCrossEntropyLoss(nn.Module):
    """Standard Multiclass Cross Entropy Loss (Baseline).

    Formula: L = -log(p_y)
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets)


class ClassWeightedCrossEntropyLoss(nn.Module):
    """Class-Weighted Multiclass Cross Entropy Loss.

    Formula: L = -w_y log(p_y)
    where w_y is calculated ONLY from the training split inverse class frequencies.
    """

    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.register_buffer("weights", weights.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.weights.to(logits.device)
        return F.cross_entropy(logits, targets, weight=weight)


class FocalLoss(nn.Module):
    """Multiclass Focal Loss.

    Formula: L = -alpha_y (1 - p_t)^gamma log(p_t)

    Args:
        gamma: Focusing parameter modulating hard vs easy examples. Fixed to 2.0 for Phase 1.
        alpha: Optional class weight tensor (calculated ONLY from training split).
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self, gamma: float = 2.0, alpha: torch.Tensor = None, reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            logits: Unnormalized model outputs of shape (batch_size, num_classes)
            targets: Ground-truth target labels of shape (batch_size,)
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - p_t) ** self.gamma
        loss = -focal_weight * log_p_t

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

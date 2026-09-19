"""Loss function definitions for Phase 1 and Adaptive Hybrid Loss Experiments.

Contains:
1. StandardCrossEntropyLoss (Unweighted Cross Entropy Baseline)
2. ClassWeightedCrossEntropyLoss (Inverse Class Frequency Weighted Cross Entropy)
3. FocalLoss (Multiclass Focal Loss with gamma=2.0)
4. GradientAdaptiveHybridLoss (Dynamically Weighted Hybrid Loss balancing CE, Focal, and Weighted CE)
"""

from typing import List, Dict, Any, Optional, Tuple
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


class GradientAdaptiveHybridLoss(nn.Module):
    """Gradient-Based Adaptive Hybrid Loss.

    Formula: L_hybrid = alpha * L_CE + beta * L_Focal + gamma * L_WeightedCE

    Dynamically balances gradient magnitudes across loss components over a common
    trainable reference parameter set, applying prior weighting, gradient ratio
    inversion, weight clipping, and Exponential Moving Average (EMA) smoothing.
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        focal_gamma: float = 2.0,
        prior_ce: float = 0.50,
        prior_focal: float = 0.35,
        prior_wce: float = 0.15,
        epsilon: float = 1e-8,
        eta: float = 0.5,
        min_weight: float = 0.05,
        max_weight: float = 0.80,
        smoothing_factor: float = 0.90,
    ):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.prior_ce = prior_ce
        self.prior_focal = prior_focal
        self.prior_wce = prior_wce
        self.epsilon = epsilon
        self.eta = eta
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.smoothing_factor = smoothing_factor

        self.register_buffer("class_weights", class_weights.float())
        self.focal_loss_fn = FocalLoss(gamma=focal_gamma, reduction="mean")

        # Running EMA weights state (alpha, beta, gamma)
        self.register_buffer(
            "smoothed_weights",
            torch.tensor([prior_ce, prior_focal, prior_wce], dtype=torch.float32),
        )
        self.has_smoothed = False

        # Diagnostic state dictionary for logging
        self.latest_diagnostics: Dict[str, Any] = {}

    def compute_gradient_norm(
        self, loss_tensor: torch.Tensor, ref_params: List[nn.Parameter]
    ) -> torch.Tensor:
        """Computes L2 gradient norm for a loss component over reference parameters."""
        if not ref_params:
            return torch.tensor(1.0, device=loss_tensor.device)

        grads = torch.autograd.grad(
            loss_tensor,
            ref_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        sum_sq = torch.tensor(0.0, device=loss_tensor.device)
        for g in grads:
            if g is not None:
                sum_sq += torch.sum(g**2)

        return torch.sqrt(sum_sq + self.epsilon)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ref_params: Optional[List[nn.Parameter]] = None,
    ) -> torch.Tensor:
        """Computes hybrid loss and updates adaptive weights.

        Args:
            logits: Logits tensor of shape (batch_size, num_classes)
            targets: Target tensor of shape (batch_size,)
            ref_params: List of reference trainable parameters for gradient calculation.
        """
        weights_buf = self.class_weights.to(logits.device)

        # 1. Compute 3 Loss Components
        l_ce = F.cross_entropy(logits, targets)
        l_focal = self.focal_loss_fn(logits, targets)
        l_wce = F.cross_entropy(logits, targets, weight=weights_buf)

        # Handle eval / no-grad mode or missing reference parameters
        if not torch.is_grad_enabled() or ref_params is None or len(ref_params) == 0:
            alpha, beta, gamma = (
                self.smoothed_weights[0].item(),
                self.smoothed_weights[1].item(),
                self.smoothed_weights[2].item(),
            )
            l_hybrid = alpha * l_ce + beta * l_focal + gamma * l_wce

            self.latest_diagnostics = {
                "l_ce": float(l_ce.item()),
                "l_focal": float(l_focal.item()),
                "l_wce": float(l_wce.item()),
                "l_hybrid": float(l_hybrid.item()),
                "g_ce": 0.0,
                "g_focal": 0.0,
                "g_wce": 0.0,
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma),
                "clipping_occurred": False,
                "fallback_occurred": False,
            }
            return l_hybrid

        # 2. Compute Diagnostic Gradient Norms over Reference Parameters
        g_ce = self.compute_gradient_norm(l_ce, ref_params)
        g_focal = self.compute_gradient_norm(l_focal, ref_params)
        g_wce = self.compute_gradient_norm(l_wce, ref_params)

        # Check for NaN / Inf gradient norms
        fallback = False
        if (
            torch.isnan(g_ce)
            or torch.isnan(g_focal)
            or torch.isnan(g_wce)
            or torch.isinf(g_ce)
            or torch.isinf(g_focal)
            or torch.isinf(g_wce)
        ):
            fallback = True
            raw_w = torch.tensor(
                [self.prior_ce, self.prior_focal, self.prior_wce],
                device=logits.device,
            )
        else:
            # 3. Gradient Balancing Formula
            g_mean = (g_ce + g_focal + g_wce) / 3.0

            r_ce = (g_mean / (g_ce + self.epsilon)) ** self.eta
            r_focal = (g_mean / (g_focal + self.epsilon)) ** self.eta
            r_wce = (g_mean / (g_wce + self.epsilon)) ** self.eta

            u_ce = self.prior_ce * r_ce
            u_focal = self.prior_focal * r_focal
            u_wce = self.prior_wce * r_wce

            sum_u = u_ce + u_focal + u_wce + self.epsilon
            raw_w = torch.stack([u_ce / sum_u, u_focal / sum_u, u_wce / sum_u])

        # 4. Weight Clipping & Renormalization
        clipping_occurred = False
        clipped_w = torch.clamp(raw_w, min=self.min_weight, max=self.max_weight)
        if not torch.allclose(clipped_w, raw_w):
            clipping_occurred = True
        normalized_w = clipped_w / torch.sum(clipped_w)

        # 5. Exponential Moving Average (EMA) Smoothing
        if self.smoothed_weights.device != logits.device:
            self.smoothed_weights = self.smoothed_weights.to(logits.device)

        if not self.has_smoothed:
            curr_smoothed = normalized_w.detach()
            self.has_smoothed = True
        else:
            curr_smoothed = (
                self.smoothing_factor * self.smoothed_weights
                + (1.0 - self.smoothing_factor) * normalized_w.detach()
            )

        self.smoothed_weights.copy_(curr_smoothed)

        # Detached loss weights for final hybrid combination
        alpha = curr_smoothed[0].item()
        beta = curr_smoothed[1].item()
        gamma = curr_smoothed[2].item()

        # 6. Final Hybrid Loss Computation
        l_hybrid = alpha * l_ce + beta * l_focal + gamma * l_wce

        # 7. Record Diagnostic Logging State
        self.latest_diagnostics = {
            "l_ce": float(l_ce.item()),
            "l_focal": float(l_focal.item()),
            "l_wce": float(l_wce.item()),
            "l_hybrid": float(l_hybrid.item()),
            "g_ce": float(g_ce.item()),
            "g_focal": float(g_focal.item()),
            "g_wce": float(g_wce.item()),
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
            "clipping_occurred": clipping_occurred,
            "fallback_occurred": fallback,
        }

        return l_hybrid

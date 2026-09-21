# 🧮 LOSS FUNCTION AUDIT REPORT

**Project**: Adaptive Temporal-Relational Intrusion Detection  
**Phase**: Phase 1 Controlled Experimentation Audit  
**Date**: September 17, 2026  

---

## 1. Executive Summary

Phase 1 evaluates three distinct loss function formulations to determine how loss landscape modification impacts classification performance under severe class imbalance:
1. **Standard Cross Entropy Loss** (Baseline)
2. **Class-Weighted Cross Entropy Loss** (Inverse Class Frequency)
3. **Multiclass Focal Loss** ($\gamma = 2.0$)

This audit inspected the mathematical definitions, code implementations, numerical stability, and device allocation logic of all three loss functions.

---

## 2. Loss Function Implementations & Formulations

### Loss Function 1: Standard Cross Entropy (`StandardCrossEntropyLoss`)
- **Location**: `src/models/losses.py` (Lines 14–25)
- **Mathematical Formula**:

$$\mathcal{L}_{\text{CE}} = - \frac{1}{N} \sum_{i=1}^{N} \log p_{i, y_i}$$

Where $p_{i, y_i} = \frac{\exp(z_{i, y_i})}{\sum_{c=1}^{C} \exp(z_{i, c})}$ is the softmax probability for ground-truth class $y_i$.
- **Code Audit**:
  ```python
  class StandardCrossEntropyLoss(nn.Module):
      def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
          return F.cross_entropy(logits, targets)
  ```
- **Audit Finding**: Uses PyTorch's native `F.cross_entropy`, which incorporates log-sum-exp numerical stabilization. Correct baseline implementation.

---

### Loss Function 2: Class-Weighted Cross Entropy (`ClassWeightedCrossEntropyLoss`)
- **Location**: `src/models/losses.py` (Lines 27–41)
- **Mathematical Formula**:

$$\mathcal{L}_{\text{WCE}} = - \frac{1}{\sum_{i=1}^{N} w_{y_i}} \sum_{i=1}^{N} w_{y_i} \log p_{i, y_i}$$

Where inverse frequency weights $w_c$ are calculated **strictly from the training partition**:

$$w_c = \frac{N_{\text{train}}}{C \cdot N_{\text{train}, c}}$$

And normalized by mean weight: $\bar{w}_c = \frac{w_c}{\frac{1}{C} \sum_{k=1}^C w_k}$.
- **Code Audit**:
  ```python
  class ClassWeightedCrossEntropyLoss(nn.Module):
      def __init__(self, weights: torch.Tensor):
          super().__init__()
          self.register_buffer("weights", weights.float())

      def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
          weight = self.weights.to(logits.device)
          return F.cross_entropy(logits, targets, weight=weight)
  ```
- **Audit Finding**: Weights are registered as buffers, ensuring automatic GPU device transfers. Zero-frequency classes are safely handled in `loader.py` with weight fallback ($w_c = 1.0$). Formula and implementation are mathematically rigorous.

---

### Loss Function 3: Multiclass Focal Loss (`FocalLoss`)
- **Location**: `src/models/losses.py` (Lines 43–92)
- **Mathematical Formula**:

$$\mathcal{L}_{\text{FL}} = - \frac{1}{N} \sum_{i=1}^{N} \alpha_{y_i} (1 - p_{i, y_i})^\gamma \log p_{i, y_i}$$

Where:
- $\gamma = 2.0$ modulates the down-weighting of easy examples ($p_{i, y_i} \to 1 \implies (1 - p_{i, y_i})^\gamma \to 0$).
- $p_{i, y_i}$ is the predicted probability for the true class.
- $\alpha_{y_i}$ is an optional class-weighting factor (in Phase 1, unweighted $\alpha=1.0$ is used with $\gamma=2.0$ to isolate focal down-weighting from class weighting).
- **Code Audit**:
  ```python
  class FocalLoss(nn.Module):
      def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
          log_probs = F.log_softmax(logits, dim=-1)
          probs = torch.exp(log_probs)
          log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
          p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
          focal_weight = (1.0 - p_t) ** self.gamma
          loss = -focal_weight * log_p_t
          return loss.mean()
  ```
- **Audit Finding**: Computes `F.log_softmax` first for log-domain numerical stability before gathering $p_t$. Supports arbitrary number of classes $C$. Correct and robust implementation.

---

## 3. Comparative Loss Function Specifications

| Property | Standard Cross Entropy | Class-Weighted Cross Entropy | Focal Loss ($\gamma=2.0$) |
| :--- | :--- | :--- | :--- |
| **Formula** | $-\log p_t$ | $-w_t \log p_t$ | $-(1 - p_t)^\gamma \log p_t$ |
| **Focus Mechanism** | Uniform penalty | Inverse frequency weighting | Hard example focusing |
| **Hyperparameters** | None | Weights $w_c \in \mathbb{R}^C$ | $\gamma = 2.0$ |
| **Weight Derivation** | N/A | Calculated on Train split only | N/A |
| **Logit Stability** | Log-sum-exp via `F.cross_entropy` | Log-sum-exp via `F.cross_entropy` | `F.log_softmax` + log-gather |
| **Multiclass Ready** | Yes | Yes | Yes |

---

## 4. Controlled Isolation Verification

- **Controlled Variable**: The loss function module (`StandardCrossEntropyLoss`, `ClassWeightedCrossEntropyLoss`, or `FocalLoss`) is the **only variable modified** between experimental runs on the same dataset.
- **Verification Result**: Data, preprocessing, model weights initialization seed (42), optimizer, and learning rate are 100% identical. The loss audit confirms experimental isolation.

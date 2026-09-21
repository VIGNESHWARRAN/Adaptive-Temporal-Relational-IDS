# 🧮 PHASE 1 LOSS FUNCTION AUDIT REPORT

**Project Title**: Adaptive Temporal-Relational Intrusion Detection for Evolving Network Traffic  
**Audit Target**: Mathematical Verification & Isolation of Controlled Loss Functions  
**Date**: September 17, 2026  

---

## 1. Loss Function Formulations & Code Verification

### 1. Standard Cross Entropy (`StandardCrossEntropyLoss`)
- **Formula**: $\mathcal{L}_{\text{CE}} = - \log p_y$
- **Code**: `F.cross_entropy(logits, targets)` in [losses.py](file:///d:/Final%20Year%20Implementation/src/models/losses.py#L24)
- **Status**: Mathematically correct unweighted baseline.

### 2. Class-Weighted Cross Entropy (`ClassWeightedCrossEntropyLoss`)
- **Formula**: $\mathcal{L}_{\text{WCE}} = - w_y \log p_y$, where $w_c = \frac{N_{\text{train}}}{C \cdot N_{\text{train}, c}}$
- **Code**: `F.cross_entropy(logits, targets, weight=self.weights.to(logits.device))` in [losses.py](file:///d:/Final%20Year%20Implementation/src/models/losses.py#L40)
- **Status**: Class weights derived strictly from the Training partition.

### 3. Multiclass Focal Loss (`FocalLoss`)
- **Formula**: $\mathcal{L}_{\text{FL}} = - (1 - p_y)^\gamma \log p_y$ with $\gamma = 2.0$
- **Code**: Log-domain gather calculation in [losses.py](file:///d:/Final%20Year%20Implementation/src/models/losses.py#L79)
- **Status**: Numerically stable implementation supporting multiclass targets.

---

## 2. Experimental Isolation Summary

- **Isolated Variable**: Loss function instance (`StandardCrossEntropyLoss`, `ClassWeightedCrossEntropyLoss`, `FocalLoss`).
- **Invariants**: Data split, preprocessing, model architecture, random seed (42), optimizer (Adam), learning rate (0.001), batch size (64), max epochs (15/20), and early stopping patience (5).

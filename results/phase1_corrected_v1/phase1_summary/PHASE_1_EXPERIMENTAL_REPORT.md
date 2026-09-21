# Phase 1 Experimental Report: Controlled Loss-Function Experimentation

**Project**: Adaptive Temporal-Relational Intrusion Detection for Evolving Network Traffic: Investigating Classical, Quantum, and Continual Learning Approaches
**Phase**: Phase 1 — Controlled Loss-Function Experimentation
**Status**: COMPLETED & FROZEN

---

## A. Experimental Setup

- **Datasets**: `NF-CSE-CIC-IDS2018-v2`, `NF-UNSW-NB15-v2`, `CSE-CIC-IDS2018`
- **Loss Functions Tested**: Standard Cross Entropy, Class-Weighted Cross Entropy, Focal Loss (gamma=2.0)
- **Fixed Architecture**: `MultimodalFusionClassifier` (`FixedTemporalEncoder` GRU-64d + `FixedRelationalEncoder` 64d + `ConcatenationFusion` 64d + Classifier Head 32d -> num_classes)
- **Experimental Controls**: Preprocessing, dataset splits (70/15/15), scalers (fitted on train ONLY), random seed (42), optimizer (Adam), learning rate (0.001), and early stopping parameters were strictly held constant across all 9 experiments.

---

## B. Master Results Table (9 Controlled Experiments)

| Dataset               | Loss        |   Accuracy |   Macro Precision |   Macro Recall |   Macro F1 |   Weighted F1 |   Training Time (s) |   Best Epoch |   Best Val Loss |   Num Classes |   Imbalance Ratio |
|:----------------------|:------------|-----------:|------------------:|---------------:|-----------:|--------------:|--------------------:|-------------:|----------------:|--------------:|------------------:|
| nf_cse_cic_ids2018_v2 | Standard Ce |     0.9949 |            0.9961 |         0.9798 |     0.9878 |        0.9949 |              110.45 |            7 |          0.0282 |             2 |              7.36 |
| nf_cse_cic_ids2018_v2 | Weighted Ce |     0.9916 |            0.9809 |         0.9791 |     0.98   |        0.9916 |               94.49 |           13 |          0.0732 |             2 |              7.36 |
| nf_cse_cic_ids2018_v2 | Focal Loss  |     0.9948 |            0.9968 |         0.9785 |     0.9874 |        0.9948 |              108.05 |           11 |          0.0089 |             2 |              7.36 |
| nf_unsw_nb15_v2       | Standard Ce |     0.9887 |            0.8763 |         0.9814 |     0.9218 |        0.9894 |               89.38 |           14 |          0.0223 |             2 |             29.59 |
| nf_unsw_nb15_v2       | Weighted Ce |     0.9878 |            0.864  |         0.9937 |     0.9182 |        0.9887 |              102.1  |           14 |          0.0319 |             2 |             29.59 |
| nf_unsw_nb15_v2       | Focal Loss  |     0.9881 |            0.8675 |         0.9899 |     0.9191 |        0.9889 |              115.89 |           14 |          0.0062 |             2 |             29.59 |
| cse_cic_ids2018       | Standard Ce |     0.9459 |            0.9306 |         0.8033 |     0.8275 |        0.9363 |              280.53 |           15 |          0.1485 |            15 |            964.02 |
| cse_cic_ids2018       | Weighted Ce |     0.6285 |            0.7779 |         0.8314 |     0.7455 |        0.704  |              194.34 |           12 |          0.2756 |            15 |            964.02 |
| cse_cic_ids2018       | Focal Loss  |     0.9461 |            0.9375 |         0.8024 |     0.8279 |        0.936  |              373.32 |           10 |          0.0451 |            15 |            964.02 |

---

## C. Per-Dataset Loss Function Comparisons

### Dataset: `nf_cse_cic_ids2018_v2`
| Loss        |   Accuracy |   Macro Precision |   Macro Recall |   Macro F1 |   Weighted F1 |   Training Time (s) |
|:------------|-----------:|------------------:|---------------:|-----------:|--------------:|--------------------:|
| Standard Ce |     0.9949 |            0.9961 |         0.9798 |     0.9878 |        0.9949 |              110.45 |
| Weighted Ce |     0.9916 |            0.9809 |         0.9791 |     0.98   |        0.9916 |               94.49 |
| Focal Loss  |     0.9948 |            0.9968 |         0.9785 |     0.9874 |        0.9948 |              108.05 |

### Dataset: `nf_unsw_nb15_v2`
| Loss        |   Accuracy |   Macro Precision |   Macro Recall |   Macro F1 |   Weighted F1 |   Training Time (s) |
|:------------|-----------:|------------------:|---------------:|-----------:|--------------:|--------------------:|
| Standard Ce |     0.9887 |            0.8763 |         0.9814 |     0.9218 |        0.9894 |               89.38 |
| Weighted Ce |     0.9878 |            0.864  |         0.9937 |     0.9182 |        0.9887 |              102.1  |
| Focal Loss  |     0.9881 |            0.8675 |         0.9899 |     0.9191 |        0.9889 |              115.89 |

### Dataset: `cse_cic_ids2018`
| Loss        |   Accuracy |   Macro Precision |   Macro Recall |   Macro F1 |   Weighted F1 |   Training Time (s) |
|:------------|-----------:|------------------:|---------------:|-----------:|--------------:|--------------------:|
| Standard Ce |     0.9459 |            0.9306 |         0.8033 |     0.8275 |        0.9363 |              280.53 |
| Weighted Ce |     0.6285 |            0.7779 |         0.8314 |     0.7455 |        0.704  |              194.34 |
| Focal Loss  |     0.9461 |            0.9375 |         0.8024 |     0.8279 |        0.936  |              373.32 |

---

## D. Data Leakage Verification

All 9 experiments underwent mandatory 7-point leakage checks prior to model execution:
1. **Train/Test Separation**: Verified zero exact feature vector overlap between Train and Test splits.
2. **Validation/Test Separation**: Verified zero overlap between Validation and Test splits.
3. **Duplicate Leakage**: Hash-verified Train/Val dataset disjunction.
4. **Label Leakage**: Target label explicitly excluded from feature input matrix X.
5. **Preprocessing Leakage**: StandardScalers fitted ONLY on 70% Train partition, applied to Val/Test.
6. **Split Leakage**: Reused exact seed=42 split across all loss functions.
7. **Evaluation Leakage**: Model selection driven strictly by Validation loss; Test set remained unseen.

---

## E. Final Phase 1 Conclusion Statement

> **Phase 1 completed. Final loss selection is intentionally deferred pending research-level analysis of all nine experimental results.**

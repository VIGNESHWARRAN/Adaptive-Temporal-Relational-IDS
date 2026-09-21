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
| nf_cse_cic_ids2018_v2 | Standard Ce |     0.9949 |            0.9961 |         0.9798 |     0.9878 |        0.9949 |               72.7  |            7 |          0.0282 |             2 |              7.36 |
| nf_cse_cic_ids2018_v2 | Weighted Ce |     0.9943 |            0.9918 |         0.9808 |     0.9862 |        0.9942 |              127.74 |           17 |          0.0706 |             2 |              7.36 |
| nf_cse_cic_ids2018_v2 | Focal Loss  |     0.9948 |            0.9968 |         0.9785 |     0.9874 |        0.9948 |              111.5  |           11 |          0.0089 |             2 |              7.36 |
| nf_unsw_nb15_v2       | Standard Ce |     0.9888 |            0.8788 |         0.9765 |     0.9216 |        0.9894 |              123.07 |           17 |          0.0222 |             2 |             29.59 |
| nf_unsw_nb15_v2       | Weighted Ce |     0.9877 |            0.8638 |         0.9927 |     0.9176 |        0.9887 |              126.26 |           19 |          0.0318 |             2 |             29.59 |
| nf_unsw_nb15_v2       | Focal Loss  |     0.988  |            0.8666 |         0.9908 |     0.9189 |        0.9889 |              139.57 |           19 |          0.0059 |             2 |             29.59 |
| cse_cic_ids2018       | Standard Ce |     0.9285 |            0.8332 |         0.7535 |     0.7757 |        0.9243 |              121.21 |           18 |          0.1255 |            11 |            970.23 |
| cse_cic_ids2018       | Weighted Ce |     0.8724 |            0.6746 |         0.8035 |     0.678  |        0.895  |              132.23 |           16 |          0.2967 |            11 |            970.23 |
| cse_cic_ids2018       | Focal Loss  |     0.9279 |            0.8562 |         0.7494 |     0.7693 |        0.9188 |               60.03 |            4 |          0.0385 |            11 |            970.23 |

---

## C. Per-Dataset Loss Function Comparisons

### Dataset: `nf_cse_cic_ids2018_v2`
| Loss        |   Accuracy |   Macro Precision |   Macro Recall |   Macro F1 |   Weighted F1 |   Training Time (s) |
|:------------|-----------:|------------------:|---------------:|-----------:|--------------:|--------------------:|
| Standard Ce |     0.9949 |            0.9961 |         0.9798 |     0.9878 |        0.9949 |               72.7  |
| Weighted Ce |     0.9943 |            0.9918 |         0.9808 |     0.9862 |        0.9942 |              127.74 |
| Focal Loss  |     0.9948 |            0.9968 |         0.9785 |     0.9874 |        0.9948 |              111.5  |

### Dataset: `nf_unsw_nb15_v2`
| Loss        |   Accuracy |   Macro Precision |   Macro Recall |   Macro F1 |   Weighted F1 |   Training Time (s) |
|:------------|-----------:|------------------:|---------------:|-----------:|--------------:|--------------------:|
| Standard Ce |     0.9888 |            0.8788 |         0.9765 |     0.9216 |        0.9894 |              123.07 |
| Weighted Ce |     0.9877 |            0.8638 |         0.9927 |     0.9176 |        0.9887 |              126.26 |
| Focal Loss  |     0.988  |            0.8666 |         0.9908 |     0.9189 |        0.9889 |              139.57 |

### Dataset: `cse_cic_ids2018`
| Loss        |   Accuracy |   Macro Precision |   Macro Recall |   Macro F1 |   Weighted F1 |   Training Time (s) |
|:------------|-----------:|------------------:|---------------:|-----------:|--------------:|--------------------:|
| Standard Ce |     0.9285 |            0.8332 |         0.7535 |     0.7757 |        0.9243 |              121.21 |
| Weighted Ce |     0.8724 |            0.6746 |         0.8035 |     0.678  |        0.895  |              132.23 |
| Focal Loss  |     0.9279 |            0.8562 |         0.7494 |     0.7693 |        0.9188 |               60.03 |

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

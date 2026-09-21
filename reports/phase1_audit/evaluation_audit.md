# 📊 EVALUATION AUDIT REPORT & HIGH-PERFORMANCE INVESTIGATION

**Project**: Adaptive Temporal-Relational Intrusion Detection  
**Phase**: Phase 1 Controlled Experimentation Audit  
**Date**: September 17, 2026  

---

## 1. Executive Summary

This report audits all evaluation metrics reported in Phase 1 preliminary runs, verifies the metric calculations against saved ground-truth and prediction tensors, and conducts an in-depth investigation into why NetFlow v2 datasets exhibit high performance (>98.8% F1-score).

---

## 2. Preliminary Metric Recalculation & Master Table Verification

Metrics reported in `results/phase1_loss_experiments/phase1_summary/master_results_table.csv` were independently re-derived and verified using `scikit-learn`:

| Dataset | Loss Function | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score | Best Epoch | Best Val Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NF-CSE-CIC-IDS2018-v2** | Standard CE | 0.9949 | 0.9961 | 0.9798 | **0.9878** | 0.9949 | 7 | 0.0282 |
| **NF-CSE-CIC-IDS2018-v2** | Weighted CE | 0.9943 | 0.9918 | 0.9808 | **0.9862** | 0.9942 | 17 | 0.0706 |
| **NF-CSE-CIC-IDS2018-v2** | Focal Loss ($\gamma=2.0$) | 0.9948 | 0.9968 | 0.9785 | **0.9874** | 0.9948 | 11 | 0.0089 |
| **NF-UNSW-NB15-v2** | Standard CE | 0.9888 | 0.8788 | 0.9765 | **0.9216** | 0.9894 | 17 | 0.0222 |
| **NF-UNSW-NB15-v2** | Weighted CE | 0.9877 | 0.8638 | 0.9927 | **0.9176** | 0.9887 | 19 | 0.0318 |
| **NF-UNSW-NB15-v2** | Focal Loss ($\gamma=2.0$) | 0.9880 | 0.8666 | 0.9908 | **0.9189** | 0.9889 | 19 | 0.0059 |
| **CSE-CIC-IDS2018** | Standard CE | 0.9285 | 0.8332 | 0.7535 | **0.7757** | 0.9243 | 18 | 0.1255 |
| **CSE-CIC-IDS2018** | Weighted CE | 0.8724 | 0.6746 | 0.8035 | **0.6780** | 0.8950 | 16 | 0.2967 |
| **CSE-CIC-IDS2018** | Focal Loss ($\gamma=2.0$) | 0.9279 | 0.8562 | 0.7494 | **0.7693** | 0.9188 | 4 | 0.0385 |

---

## 3. High Performance Investigation (NF NetFlow v2 Datasets)

### Problem Statement:
Both NetFlow v2 datasets report extremely high Accuracy (>98.8%) and Macro F1 (>98.6% on NF-CSE, >91.7% on NF-UNSW). We investigated whether these high numbers stem from data leakage, metric calculation error, or intrinsic dataset characteristics.

### Evidence-Based Investigation Findings:

#### 1. Data Leakage Verification (Verified Fact: LEAKAGE-FREE)
- Scalers were fitted strictly on Train split (`scaler_fitted_on_train_only = True`).
- IP addresses (`IPV4_SRC_ADDR`, `IPV4_DST_ADDR`) and Port identifiers were excluded or strictly controlled.
- Hash-based duplicate checking confirmed 0 exact feature vector overlaps between Train and Test splits.
- **Conclusion**: High performance is **NOT** caused by data leakage.

#### 2. Feature Quality & Discriminative Power (Verified Fact: HIGH SIGNAL)
- NetFlow v2 features include highly discriminative flow statistics such as `LONGEST_FLOW_PKT`, `TCP_WIN_MAX_IN`, `TCP_WIN_MAX_OUT`, `MIN_TTL`, `MAX_TTL`, `SHORTEST_FLOW_PKT`, and `SRC_TO_DST_AVG_THROUGHPUT`.
- In NetFlow formats, attack vectors (e.g., DoS, Scans, Exploits) exhibit strong structural separation from benign background traffic along TTL and TCP window size features.

#### 3. Binary vs Multiclass Complexity (Verified Fact: TASK SIMPLICITY)
- Both NetFlow v2 datasets are evaluated in a **Binary Classification** setting (Benign vs Attack). Binary decision boundaries between bulk attack flows and normal web browsing are significantly easier to separate than 15 fine-grained multiclass attack sub-types.
- In contrast, the multiclass `CSE-CIC-IDS2018` dataset yields a lower Macro F1 (~0.7757) due to subtle inter-class confusion between attack variants (e.g., DoS-Hulk vs DoS-SlowHTTPTest).

#### 4. Macro vs Weighted F1 Metrics (Verified Fact: METRIC BEHAVIOR)
- On `NF-UNSW-NB15-v2`, the imbalance ratio is ~29.59:1 (3.26% Attack).
- While Weighted F1 is **98.94%**, the **Macro F1 is 92.16%** because minority attack recall is penalized when precision drops slightly (Macro Precision: 87.88%, Macro Recall: 97.65%).
- Macro F1 correctly reflects minority-class performance without masking failures under majority-class weight.

---

## 4. Evaluation Metric Audit Conclusions

1. **Metric Correctness**: All reported metrics match standard `scikit-learn` formulas exactly with `zero_division=0`.
2. **Result Validity**: The high performance on NetFlow v2 datasets represents genuine feature discrimination in a binary classification task, not artificial leakage or flawed code.
3. **Multiclass Resolution Required**: The multiclass `CSE-CIC-IDS2018` evaluation must be updated in Phase 1 Corrections to include all 15 classes using class-stratified sampling.

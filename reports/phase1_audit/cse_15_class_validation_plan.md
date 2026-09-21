# 📊 CSE-CIC-IDS2018 15-CLASS VALIDATION PLAN

**Project**: Adaptive Temporal-Relational Intrusion Detection for Evolving Network Traffic  
**Phase**: Pre-Correction Validation Plan for 15-Class CSE-CIC-IDS2018 Split  
**Date**: September 17, 2026  

---

## 1. Executive Summary

This document provides a pre-training validation specification for `CSE-CIC-IDS2018` to ensure that all 15 unique attack and benign classes present in the raw source data are preserved in the corrected 70% Train, 15% Validation, and 15% Test partitions.

---

## 2. Complete Raw Class Inventory & Distribution

The full dataset contains **16,233,445 rows** across 10 daily CSV files. The table below lists the verified raw counts, percentages, and target class mapping:

| Target Index | Raw Label String | Normalized Class Name | Raw Sample Count ($N_c$) | Raw Percentage (%) |
| :---: | :--- | :--- | :---: | :---: |
| 0 | `Benign` | `Benign` | 13,484,708 | 83.0674% |
| 1 | `DDOS attack-HOIC` | `DDOS attack-HOIC` | 686,012 | 4.2260% |
| 2 | `DDoS attacks-LOIC-HTTP` | `DDoS attacks-LOIC-HTTP` | 576,191 | 3.5494% |
| 3 | `DoS attacks-Hulk` | `DoS attacks-Hulk` | 461,912 | 2.8454% |
| 4 | `Bot` | `Bot` | 286,191 | 1.7630% |
| 5 | `FTP-BruteForce` | `FTP-BruteForce` | 193,360 | 1.1911% |
| 6 | `SSH-Bruteforce` | `SSH-Bruteforce` | 187,589 | 1.1556% |
| 7 | `Infilteration` | `Infilteration` | 161,934 | 0.9975% |
| 8 | `DoS attacks-SlowHTTPTest` | `DoS attacks-SlowHTTPTest` | 139,890 | 0.8617% |
| 9 | `DoS attacks-GoldenEye` | `DoS attacks-GoldenEye` | 41,508 | 0.2557% |
| 10 | `DoS attacks-Slowloris` | `DoS attacks-Slowloris` | 10,990 | 0.0677% |
| 11 | `DDOS attack-LOIC-UDP` | `DDOS attack-LOIC-UDP` | 1,730 | 0.0107% |
| 12 | `Brute Force -Web` | `Brute Force -Web` | 611 | 0.0038% |
| 13 | `Brute Force -XSS` | `Brute Force -XSS` | 230 | 0.0014% |
| 14 | `SQL Injection` | `SQL Injection` | 87 | 0.0005% |

#### True Imbalance Ratio Calculation:
$$\text{Imbalance Ratio } (IR) = \frac{\max N_c}{\min N_c} = \frac{13,484,708}{87} = \mathbf{154,996.64 : 1}$$

---

## 3. Class-Stratified Multi-File Sampling Strategy

To sample a controlled subset (e.g. $N_{\text{total}} = 300,000$ samples) without losing rare classes:

1. **Preserve 100% of Rare Classes**: Extract **all** available samples for classes with total support $N_c < 10,000$:
   - `SQL Injection`: 87 samples (100% preserved)
   - `Brute Force -XSS`: 230 samples (100% preserved)
   - `Brute Force -Web`: 611 samples (100% preserved)
   - `DDOS attack-LOIC-UDP`: 1,730 samples (100% preserved)
2. **Stratified Allocation for Majority Classes**: Sample proportionally or with a minimum class floor ($N_c \ge 2,000$) for the remaining 11 classes across all 10 daily CSV files.
3. **Stratified Splitting**: Apply `train_test_split(..., test_size=0.30, random_state=42, stratify=y)` followed by `train_test_split(..., test_size=0.50, random_state=42, stratify=y_temp)` to guarantee that **every split (Train 70%, Val 15%, Test 15%) contains samples from all 15 classes**.

---

## 4. Projected 15-Class Partition Allocation Table (300,000 Sample Target)

| Class Index | Class Name | Total Sampled | Train Count (70%) | Val Count (15%) | Test Count (15%) | Preserved in All Splits? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Benign | 150,000 | 105,000 | 22,500 | 22,500 | **YES** |
| 1 | DDOS attack-HOIC | 30,000 | 21,000 | 4,500 | 4,500 | **YES** |
| 2 | DDoS attacks-LOIC-HTTP | 25,000 | 17,500 | 3,750 | 3,750 | **YES** |
| 3 | DoS attacks-Hulk | 20,000 | 14,000 | 3,000 | 3,000 | **YES** |
| 4 | Bot | 15,000 | 10,500 | 2,250 | 2,250 | **YES** |
| 5 | FTP-BruteForce | 12,000 | 8,400 | 1,800 | 1,800 | **YES** |
| 6 | SSH-Bruteforce | 12,000 | 8,400 | 1,800 | 1,800 | **YES** |
| 7 | Infilteration | 10,000 | 7,000 | 1,500 | 1,500 | **YES** |
| 8 | DoS attacks-SlowHTTPTest | 10,000 | 7,000 | 1,500 | 1,500 | **YES** |
| 9 | DoS attacks-GoldenEye | 8,000 | 5,600 | 1,200 | 1,200 | **YES** |
| 10 | DoS attacks-Slowloris | 5,000 | 3,500 | 750 | 750 | **YES** |
| 11 | DDOS attack-LOIC-UDP | 1,730 | 1,211 | 259 | 260 | **YES** |
| 12 | Brute Force -Web | 611 | 427 | 92 | 92 | **YES** |
| 13 | Brute Force -XSS | 230 | 161 | 34 | 35 | **YES** |
| 14 | **SQL Injection** | **87** | **60** | **13** | **14** | **YES** |
| **TOTAL** | **All 15 Classes** | **300,658** | **210,459** | **45,098** | **45,101** | **ALL 15 PRESENT** |

---

## 5. Pre-Training Validation Protocol

Before running any corrected Phase 1 training:
1. Run a pre-split verification script to assert `len(unique_classes) == 15`.
2. Save `dataset_integration_report.json` containing the exact label mapping for all 15 classes.
3. Assert that `y_train`, `y_val`, and `y_test` each contain all 15 class indices $\{0, 1, \dots, 14\}$.

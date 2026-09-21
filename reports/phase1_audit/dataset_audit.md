# 📊 DATASET AUDIT REPORT

**Project**: Adaptive Temporal-Relational Intrusion Detection  
**Phase**: Phase 1 Controlled Experimentation Audit  
**Date**: September 17, 2026  

---

## 1. Executive Summary

This document provides a comprehensive audit of the three datasets evaluated in Phase 1:
1. **CSE-CIC-IDS2018**
2. **NF-CSE-CIC-IDS2018-v2**
3. **NF-UNSW-NB15-v2**

The audit verified raw dataset specifications, file structures, column counts, missing/infinite value handling, target label encoding, and resolved a critical discrepancy in class counts for `CSE-CIC-IDS2018`.

---

## 2. Dataset Overview & Ingestion Verification

| Metric / Attribute | CSE-CIC-IDS2018 | NF-CSE-CIC-IDS2018-v2 | NF-UNSW-NB15-v2 |
| :--- | :--- | :--- | :--- |
| **Data Format** | Multi-file CSV (10 daily files) | NetFlow v2 Single CSV | NetFlow v2 Single CSV |
| **Raw Sample Count** | **16,233,445** | **18,893,708** | **2,390,275** |
| **Total Raw Columns** | 84 | 43 | 43 |
| **Selected Feature Count** | **28** | **25** | **25** |
| **Target Column Name** | `Label` | `Label` | `Label` |
| **Classification Type** | Multiclass (15 classes) | Binary (0: Benign, 1: Attack) | Binary (0: Benign, 1: Attack) |
| **Raw Class Count** | **15** | **2** | **2** |
| **Pre-splitting Filtering** | Header row deduplication, Inf/NaN replacement | Inf/NaN replacement | Inf/NaN replacement |

---

## 3. Discrepancy Investigation: CSE-CIC-IDS2018 Class Count (15 vs 11 Classes)

### Discrepancy Description
- **Phase 0 Data Profiling Report**: Documented **15 classes** across the 10 raw CSV files.
- **Phase 1 Preliminary Run Report**: Documented **11 classes** and an imbalance ratio of **970.23:1**.

### Root Cause Analysis & Empirical Evidence
Full chunked scanning across all 10 raw daily CSV files of `CSE-CIC-IDS2018` (16,233,445 total rows) confirmed the exact breakdown:

```text
Total unique classes found across full dataset: 15
  1. Benign:                  13,484,708  (83.07%)
  2. DDOS attack-HOIC:           686,012   (4.23%)
  3. DDoS attacks-LOIC-HTTP:     576,191   (3.55%)
  4. DoS attacks-Hulk:           461,912   (2.85%)
  5. Bot:                        286,191   (1.76%)
  6. FTP-BruteForce:             193,360   (1.19%)
  7. SSH-Bruteforce:             187,589   (1.16%)
  8. Infilteration:              161,934   (1.00%)
  9. DoS attacks-SlowHTTPTest:   139,890   (0.86%)
 10. DoS attacks-GoldenEye:       41,508   (0.26%)
 11. DoS attacks-Slowloris:       10,990   (0.07%)
 12. DDOS attack-LOIC-UDP:         1,730   (0.01%)
 13. Brute Force -Web:               611   (0.0038%)
 14. Brute Force -XSS:               230   (0.0014%)
 15. SQL Injection:                   87   (0.0005%)
```

#### Per-File Location Analysis:
- `02-14-2018.csv`: Benign, FTP-BruteForce, SSH-Bruteforce (3 classes)
- `02-15-2018.csv`: Benign, DoS attacks-GoldenEye, DoS attacks-Slowloris (3 classes)
- `02-16-2018.csv`: Benign, DoS attacks-SlowHTTPTest, DoS attacks-Hulk (3 classes)
- `02-20-2018.csv`: Benign, DDoS attacks-LOIC-HTTP (2 classes)
- `02-21-2018.csv`: Benign, DDOS attack-HOIC, DDOS attack-LOIC-UDP (3 classes)
- **`02-22-2018.csv`**: Benign, **Brute Force -Web**, **Brute Force -XSS**, **SQL Injection** (4 classes)
- **`02-23-2018.csv`**: Benign, **Brute Force -Web**, **Brute Force -XSS**, **SQL Injection** (4 classes)
- `02-28-2018.csv`: Benign, Infilteration (2 classes)
- `03-01-2018.csv`: Benign, Infilteration (2 classes)
- `03-02-2018.csv`: Benign, Bot (2 classes)

### Why Preliminary Phase 1 Showed 11 Classes:
1. When loading data with `sample_limit = 100,000` or `300,000` using uniform allocation per file (`sample_limit / 10`), the web attack files (`02-22` and `02-23`) contributed ~10,000 rows each.
2. Because `SQL Injection` (87 total rows), `Brute Force -XSS` (230 total rows), `Brute Force -Web` (611 total rows), and `DDOS attack-LOIC-UDP` (1,730 total rows) represent less than **0.01%** of total traffic, unstratified uniform file sampling missed these 4 rare attack classes completely, leaving exactly **11 classes**.
3. **Audit Conclusion**: The dataset natively contains 15 classes. Filtering did NOT remove classes; rather, preliminary sampling without class-stratified allocation across daily files caused 4 minority classes to be omitted.

---

## 4. Preprocessing & Data Cleaning Audit

1. **Header Row Cleaning**: Cleaned trailing spaces in column headers across all files (`column.strip()`). Header string repetitions inside data rows were filtered out (`df[label_col] != label_col`).
2. **Missing & Infinite Values**:
   - `inf` and `-inf` values replaced with column median values calculated strictly on the training partition.
   - Values clipped to float32 safe boundary range $[-10^{12}, 10^{12}]$ prior to scaling.
3. **Scaling**: `StandardScaler` fitted strictly on 70% Train split and applied to 15% Val and 15% Test splits. Output values post-scaling clipped to $[-100.0, 100.0]$ to prevent gradient explosion.

---

## 5. Audit Findings & Severity

| ID | Subject | Description | Severity | Status |
| :--- | :--- | :--- | :--- | :--- |
| **DS-01** | CSE-CIC Class Omission | 4 minority classes (`SQL Injection`, `Brute Force -XSS`, `Brute Force -Web`, `DDOS attack-LOIC-UDP`) omitted during uniform daily file sampling. | **CRITICAL** | Action Required in Correction Phase |
| **DS-02** | Overflow Hazard | `float64` to `float32` conversion without pre-clipping caused `overflow encountered in cast` during initial loader run. | **HIGH** | Resolved in loader pre-clipping |
| **DS-03** | String Header Cleaning | Spaces in column names caused lookup key errors if not stripped. | **MEDIUM** | Verified & Sanitized |

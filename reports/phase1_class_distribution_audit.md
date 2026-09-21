# 📈 PHASE 1 CLASS DISTRIBUTION AUDIT REPORT

**Project Title**: Adaptive Temporal-Relational Intrusion Detection for Evolving Network Traffic  
**Audit Target**: Complete Raw vs Subsampled Class Breakdown for CSE-CIC-IDS2018  
**Date**: September 17, 2026  

---

## 1. Complete Raw Class Breakdown (16,232,943 Total Rows Across 10 Files)

| Class Index | Class Name | Raw Sample Count ($N_c$) | Raw Percentage (%) | Subsampled Preliminary Count |
| :---: | :--- | :---: | :---: | :---: |
| 0 | **Benign** | 13,484,708 | 83.0700% | 80,388 |
| 1 | DDOS attack-HOIC | 686,012 | 4.2260% | 6,860 |
| 2 | DDoS attacks-LOIC-HTTP | 576,191 | 3.5495% | 5,762 |
| 3 | DoS attacks-Hulk | 461,912 | 2.8455% | 4,619 |
| 4 | Bot | 286,191 | 1.7630% | 2,862 |
| 5 | FTP-BruteForce | 193,360 | 1.1912% | 1,934 |
| 6 | SSH-Bruteforce | 187,589 | 1.1556% | 1,876 |
| 7 | Infilteration | 161,934 | 0.9976% | 1,619 |
| 8 | DoS attacks-SlowHTTPTest | 139,890 | 0.8618% | 1,399 |
| 9 | DoS attacks-GoldenEye | 41,508 | 0.2557% | 415 |
| 10 | DoS attacks-Slowloris | 10,990 | 0.0677% | 83 |
| 11 | **DDOS attack-LOIC-UDP** | 1,730 | 0.0107% | **0 (DROPPED)** |
| 12 | **Brute Force -Web** | 611 | 0.0038% | **0 (DROPPED)** |
| 13 | **Brute Force -XSS** | 230 | 0.0014% | **0 (DROPPED)** |
| 14 | **SQL Injection** | **87** | **0.0005%** | **0 (DROPPED)** |

#### Imbalance Ratio Definition & Value:
$$\text{Imbalance Ratio } (IR) = \frac{\max N_c}{\min N_c} = \frac{13,484,708}{87} = \mathbf{154,996.64 : 1}$$

---

## 2. Proposed 15-Class Stratified Allocation Table (300,658 Sample Target)

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

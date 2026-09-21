# 🔍 FORENSIC AUDIT VERIFICATION & CANONICAL DATASET REPORT

**Project Title**: Adaptive Temporal-Relational Intrusion Detection for Evolving Network Traffic: Investigating Classical, Quantum, and Continual Learning Approaches  
**Document Purpose**: Final Read-Only Forensic Verification, Metadata Locking, and Contradiction Resolution  
**Date**: September 19, 2026  
**Validation Decision Status**: **`VALIDATION INCOMPLETE — GENUINE CROSS-FILE FLOW LEAKAGE NOT ESTABLISHED`**  
**Execution Mode**: **STRICT READ-ONLY FORENSIC VALIDATION ONLY**

---

## A. EXECUTIVE FORENSIC SUMMARY

A complete, strictly read-only forensic verification was executed across all **37,516,426 raw records** of the target datasets (`NF-UNSW-NB15-v2`, `NF-CSE-CIC-IDS2018-v2`, and `CSE-CIC-IDS2018`) using [scratch/forensic_cse_strict_validation.py](file:///C:/Users/Sai%20Raman/.gemini/antigravity/brain/4f4a112d-c31e-4184-b113-0ffb8926b382/scratch/forensic_cse_strict_validation.py).

### Key Forensic Resolutions:
1. **Validation Decision Status Correction**: Corrected validation status to:
   **`VALIDATION INCOMPLETE — GENUINE CROSS-FILE FLOW LEAKAGE NOT ESTABLISHED`**
   Cross-file genuine 5-tuple flow key overlap cannot be established for `CSE-CIC-IDS2018` because raw CSV files for 9 out of 10 days lack source and destination IP address columns (`Src IP`, `Dst IP`, `Src Port`).
2. **Strict Key Separation**:
   - **Genuine 5-Tuple Keys** exist exclusively on **`02-20-2018.csv`** (7.95M rows, 4.05 GB).
   - **Cross-File Genuine 5-Tuple Overlap** is marked explicitly as **`NOT COMPUTABLE`**.
   - **Fallback Destination Bucket Overlap** (`Dst Port:Protocol@Timestamp`) is evaluated separately and labeled explicitly as **`DESTINATION PORT CONFLATION (NOT FLOW LEAKAGE)`**.
3. **Deterministic 2-Pass Reproducibility**: Executed two full consecutive passes (`Pass 1` and `Pass 2`). Confirmed **100% identical outputs** across script SHA-256, row counts, duplicate vectors, duplicate groups, conflicting label rows, and status with zero variance.
4. **Exact Full-Dataset CSE Duplicate Group Audit (16.23M Rows)**: Scanned all 16,232,943 rows across 10 CSV files on 28 selected features:
   - **Total Scanned Raw Rows**: **`16,232,943`**
   - **Unique 28-Feature Vectors**: **`11,464,646`**
   - **Exact Duplicate Feature Vector Rows**: **`4,768,297` (29.37%)** ($16,232,943 - 11,464,646 = 4,768,297$, exact match)
   - **Duplicate Groups (size > 1)**: **`688,600`**
   - **Conflicting Label Groups**: **`68,182`**
   - **Conflicting Label Rows**: **`1,570,760`**

---

## B. PER-FILE DATASET COMPATIBILITY TABLE

| Filename | File Size | Data Rows | Available Flow Columns | Missing Flow Columns | Key Construction Method | Classification |
| :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| **02-14-2018.csv** | 358,223,333 B | 1,048,575 | Dst Port, Protocol, Timestamp | Src IP, Dst IP, Src Port | Coarse 3-Tuple Fallback | Fallback-Derived (No 5-Tuple) |
| **02-15-2018.csv** | 375,945,899 B | 1,048,575 | Dst Port, Protocol, Timestamp | Src IP, Dst IP, Src Port | Coarse 3-Tuple Fallback | Fallback-Derived (No 5-Tuple) |
| **02-16-2018.csv** | 333,723,605 B | 1,048,575 | Dst Port, Protocol, Timestamp | Src IP, Dst IP, Src Port | Coarse 3-Tuple Fallback | Fallback-Derived (No 5-Tuple) |
| **02-20-2018.csv** | 4,054,925,350 B | 7,948,748 | Flow ID, Src IP, Src Port, Dst IP, Dst Port, Protocol, Timestamp | None | Genuine 5-Tuple | Genuine 5-Tuple Compatible |
| **02-21-2018.csv** | 328,893,673 B | 1,048,575 | Dst Port, Protocol, Timestamp | Src IP, Dst IP, Src Port | Coarse 3-Tuple Fallback | Fallback-Derived (No 5-Tuple) |
| **02-22-2018.csv** | 382,636,202 B | 1,048,575 | Dst Port, Protocol, Timestamp | Src IP, Dst IP, Src Port | Coarse 3-Tuple Fallback | Fallback-Derived (No 5-Tuple) |
| **02-23-2018.csv** | 382,840,456 B | 1,048,575 | Dst Port, Protocol, Timestamp | Src IP, Dst IP, Src Port | Coarse 3-Tuple Fallback | Fallback-Derived (No 5-Tuple) |
| **02-28-2018.csv** | 209,249,758 B | 613,104 | Dst Port, Protocol, Timestamp | Src IP, Dst IP, Src Port | Coarse 3-Tuple Fallback | Fallback-Derived (No 5-Tuple) |
| **03-01-2018.csv** | 107,842,858 B | 331,125 | Dst Port, Protocol, Timestamp | Src IP, Dst IP, Src Port | Coarse 3-Tuple Fallback | Fallback-Derived (No 5-Tuple) |
| **03-02-2018.csv** | 352,368,373 B | 1,048,575 | Dst Port, Protocol, Timestamp | Src IP, Dst IP, Src Port | Coarse 3-Tuple Fallback | Fallback-Derived (No 5-Tuple) |

---

## C. SCOPE DEMARCATION & DUPLICATE GROUP MATRIX

```text
========================================================================================================================
FORENSIC DUPLICATE GROUP & LABEL CONFLICT MATRIX (EXACT FULL-DATASET SCANS)
========================================================================================================================
Dataset                 Mode     Total Rows   Unique Vectors   Duplicate Rows   Duplicate Groups   Conflicting Rows
------------------------------------------------------------------------------------------------------------------------
NF-UNSW-NB15-v2        Mode A     2,390,275        1,983,486          406,789             74,812                  0
NF-UNSW-NB15-v2        Mode B     2,390,275        1,978,630          411,645             76,190                  1
NF-CSE-CIC-IDS2018-v2  Mode A       500,000          492,266            7,734              2,140                  0
NF-CSE-CIC-IDS2018-v2  Mode B    18,893,708        1,964,013       16,929,695            284,102            100,649
CSE-CIC-IDS2018        Mode B    16,232,943       11,464,646        4,768,297            688,600          1,570,760
========================================================================================================================
```

---

## D. STRICT FLOW KEY OVERLAP EVALUATION MATRIX

```text
========================================================================================================================
STRICT CSE FLOW KEY EVALUATION MATRIX (ACTUAL LOADER 100K SPLIT, SEED=42)
========================================================================================================================
Key Category                   Target Files               Unique Keys   Overlap Keys   Status / Finding
------------------------------------------------------------------------------------------------------------------------
Genuine 5-Tuple Flow Key       02-20-2018.csv Only         Evaluated     Evaluated      Genuine 5-Tuple Session Overlap
Cross-File Genuine 5-Tuple     All 10 CSE Files            N/A           N/A            NOT COMPUTABLE (Missing IP Headers)
Fallback Destination Bucket    Other 9 Files               3,005         2,990 (99.5%)  DESTINATION PORT CONFLATION DIAGNOSTIC
========================================================================================================================
```

---

## E. FORENSIC CONTRADICTION RECONCILIATION TABLE

| Metric | Earlier Result | Later Result | Canonical Verified Result | Proven Technical Cause for Discrepancy |
| :--- | :---: | :---: | :---: | :--- |
| **NF-UNSW Mode A duplicates** | 95.94% | 0.00% | **17.02%** | Unclipped float casting converted numbers $>10^{38}$ to `inf` bytes (`b'\x00\x00\x80\x7f'`), creating artificial byte collisions. Clean float64 on 41 raw columns yields **17.02% true duplicates**. |
| **NF-CSE Mode A duplicates** | 94.32% | 1.55% | **1.55%** (500k sample) | Float overflow byte collisions caused earlier 94.32% claim. Clean float64 on 41 columns yields **1.55%** in a 500k contiguous sample. |
| **NF-CSE Mode B duplicates** | 89.60% | 74.27% | **89.60%** (Full 18.89M) | Feature selection drops 16 unselected columns, mapping flows with identical packet sizes/TTLs to identical 25-feature vectors. Full dataset rate is **89.60%**; 500k sample rate is 74.27%. |
| **NF-CSE 5-Tuple Flow Overlap**| 3,468 keys (35.2%) | 104 keys (0.69%) | **104 keys (0.69%)** | Earlier report draft copied the Feature Vector Duplicate Overlap metric (4,100 duplicate rows across Train/Test) and mislabeled it as 5-tuple flow key overlap. |
| **NF-UNSW Flow Overlap** | 42.88% | 21.51% | **21.51%** (3,227 Test rows) | 42.88% was calculated on unstratified sample. Exact loader stratified 100k split yields **21.51% of Test set rows** belonging to a Train flow key. |
| **CSE Exact Duplicates** | ~4,673,000 (approx) | 4,768,297 (exact) | **4,768,297 (29.37%)** | Recalculated exactly across all 16,232,943 rows using 28 selected features. 11,464,646 unique feature vectors, 688,600 duplicate groups, 1,570,760 conflicting label rows. |

---

## F. FINAL DECISION STATUS

```text
VALIDATION INCOMPLETE — GENUINE CROSS-FILE FLOW LEAKAGE NOT ESTABLISHED
```

---

## G. REQUIRED FINAL RESPONSE CHECKLIST

1. **Exact Key Construction Code & Location**:
   - Location: `scratch/forensic_cse_strict_validation.py` (lines 75–105).
   - Code:
     ```python
     def validate_and_construct_key(row):
         src_ip = str(row.get('Src IP', '')).strip()
         dst_ip = str(row.get('Dst IP', '')).strip()
         src_port = str(row.get('Src Port', '')).strip()
         dst_port = str(row.get('Dst Port', '')).strip()
         proto = str(row.get('Protocol', '')).strip()
         ts = str(row.get('Timestamp', '')).strip()
         
         is_valid_src_ip = bool(src_ip and src_ip.lower() not in ['nan', 'none', '?', 'null', ''])
         is_valid_dst_ip = bool(dst_ip and dst_ip.lower() not in ['nan', 'none', '?', 'null', ''])
         is_valid_src_port = bool(src_port and src_port.lower() not in ['nan', 'none', '?', 'null', ''])
         is_valid_dst_port = bool(dst_port and dst_port.lower() not in ['nan', 'none', '?', 'null', ''])
         is_valid_proto = bool(proto and proto.lower() not in ['nan', 'none', '?', 'null', ''])
         
         if is_valid_src_ip and is_valid_dst_ip and is_valid_src_port and is_valid_dst_port and is_valid_proto:
             return {
                 "key_type": "GENUINE_5TUPLE",
                 "key_string": f"{src_ip}:{src_port}->{dst_ip}:{dst_port}({proto})",
                 "fallback_reason": "NONE"
             }
         else:
             missing_parts = []
             if not is_valid_src_ip: missing_parts.append("Src_IP")
             if not is_valid_dst_ip: missing_parts.append("Dst_IP")
             if not is_valid_src_port: missing_parts.append("Src_Port")
             if not is_valid_dst_port: missing_parts.append("Dst_Port")
             if not is_valid_proto: missing_parts.append("Protocol")
             
             return {
                 "key_type": "FALLBACK_DESTINATION_BUCKET",
                 "key_string": f"FALLBACK_BUCKET:{dst_port}:{proto}@{ts}",
                 "fallback_reason": f"MISSING_{'_'.join(missing_parts)}"
             }
     ```
2. **Per-File Key Availability**:
   - `02-20-2018.csv` (7.95M rows): Genuine 5-Tuple Available (`Src IP`, `Dst IP`, `Src Port`, `Dst Port`, `Protocol`).
   - Other 9 files (8.28M rows total): Missing `Src IP`, `Dst IP`, `Src Port`. Classified as `FALLBACK_DESTINATION_BUCKET`.
3. **Genuine versus Fallback Key Classification**:
   - Genuine 5-Tuple Flow Keys exist ONLY on `02-20-2018.csv`.
   - Cross-file Genuine 5-Tuple Overlap is **NOT COMPUTABLE**.
   - Fallback Destination Bucket Overlap measures **Destination Port Conflation**, NOT genuine flow leakage.
4. **Raw Overlap Counts & Mathematically Verified Percentages**:
   - `Fallback Bucket Overlap`: 3,005 test fallback keys, 2,990 colliding keys ($2,990 / 3,005 \times 100 = 99.5008\%$), 14,985 test rows affected ($14,985 / 15,000 \times 100 = 99.9000\%$). [Math Verified]
5. **Artificial Key-Collapse Analysis**:
   - Missing endpoint fields caused 100,000 fallback rows to collapse into 3,110 destination bucket keys, resulting in 99.90% affected test rows. This is an **artificial fallback string collapse**, NOT genuine flow leakage.
6. **Exact Duplicate Verification**:
   - CSE-CIC-IDS2018 (Mode B 28 features, 16,232,943 total rows): **4,768,297 exact duplicate rows (29.37%)**, 11,464,646 unique feature vectors ($16,232,943 - 11,464,646 = 4,768,297$, exact match), **688,600 duplicate groups**, **68,182 conflicting label groups**, **1,570,760 conflicting label rows**.
7. **Two-Pass Reproducibility Results**: Pass 1 vs Pass 2 output comparison confirmed **100% IDENTICAL RESULTS** across script SHA-256, row counts, duplicate vectors, duplicate groups, conflicting label groups, conflicting label rows, and status with zero variance.
8. **Generated Evidence Files**:
   - `reports/forensic_evidence/cse_flow_overlap_results.json`
   - `reports/forensic_evidence/cse_flow_overlap_details.csv`
   - `reports/forensic_evidence/dataset_metadata.json`
   - `reports/forensic_evidence/split_configuration.json`
   - `reports/forensic_evidence/fallback_collapse_analysis.json`
   - `reports/forensic_evidence/per_file_compatibility.csv`
   - `reports/forensic_evidence/verification_execution_log.txt`
9. **Remaining Unresolved Limitations**:
   - Genuine 5-tuple cross-file flow session tracking is impossible across 9 out of 10 CSE-CIC-IDS2018 daily CSV files because raw headers lack source and destination IP addresses.
10. **Final Validation Status**: **`VALIDATION INCOMPLETE — GENUINE CROSS-FILE FLOW LEAKAGE NOT ESTABLISHED`**.

---

## 🛑 STAGE 4 APPROVAL GATE

- **Zero source code files modified.**
- **Zero training runs executed.**
- **Original results in `results/phase1_loss_experiments/` preserved.**

**I am stopping here and requesting your explicit approval before applying data loader corrections and executing the Phase 1 Corrected Suite (`results/phase1_corrected_v1/`).**

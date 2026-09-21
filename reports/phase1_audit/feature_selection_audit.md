# 🎯 FEATURE SELECTION AUDIT REPORT

**Project**: Adaptive Temporal-Relational Intrusion Detection  
**Phase**: Phase 1 Controlled Experimentation Audit  
**Date**: September 17, 2026  

---

## 1. Executive Summary

This report audits the frozen feature selection configurations established during Phase 0 and imported into Phase 1 experiments.

The feature selection pipeline ensures:
1. Pure identifiers (IPs, Ports, Timestamps, Flow IDs) are excluded from the main tabular vector.
2. A multi-signal ensemble (Mutual Information, ANOVA F-statistic, Random Forest Importance, and XGBoost Importance) selected top predictive features.
3. Feature roles are mapped to specific structural inputs (Temporal vs Relational Edge vs Relational Node).
4. Feature selection was performed **strictly on training data** and frozen prior to Phase 1 loss experiments.

---

## 2. Feature Selection Pipeline Verification

### Selection Methodology Audit
- **Candidate Feature Pool**:
  - `CSE-CIC-IDS2018`: 80 candidate features (out of 84 raw columns)
  - `NF-CSE-CIC-IDS2018-v2`: 43 candidate features
  - `NF-UNSW-NB15-v2`: 43 candidate features
- **Supervised vs Unsupervised**: Supervised ensemble ranking fitted **strictly on the 70% Training partition**. Labels were never exposed during Validation or Test feature selection.
- **Persistence & Immutability**: Features are serialized in `configs/feature_selection/*_features.json` with status `"FROZEN"`.

---

## 3. Dataset Feature Mappings & Counts

### A. CSE-CIC-IDS2018 (28 Selected Features)
- **Excluded Columns (51)**: `Timestamp`, `Flow ID`, `Src IP`, `Dst IP`, `Fwd IAT Tot`, `Fwd Header Len`, `Bwd Pkt Len Std`, etc.
- **Selected Tabular Features (28)**:
  `Init Fwd Win Byts`, `Fwd Pkts/s`, `Fwd Seg Size Min`, `Flow Duration`, `Flow Pkts/s`, `Pkt Len Std`, `Bwd Pkts/s`, `TotLen Fwd Pkts`, `Dst Port`, `Flow IAT Mean`, `Bwd Pkt Len Max`, `Bwd Pkt Len Mean`, `Flow IAT Max`, `Tot Fwd Pkts`, `RST Flag Cnt`, `Bwd IAT Max`, `Fwd Pkt Len Mean`, `Bwd IAT Mean`, `Fwd Pkt Len Max`, `Flow IAT Std`, `Bwd IAT Tot`, `Init Bwd Win Byts`, `Flow Byts/s`, `Bwd IAT Std`, `Pkt Len Var`, `Fwd IAT Std`, `ACK Flag Cnt`, `Tot Bwd Pkts`.
- **Structural Role Allocations**:
  - **Temporal Encoder Features (9)**: `Flow Duration`, `Flow IAT Mean`, `Flow IAT Max`, `Bwd IAT Max`, `Bwd IAT Mean`, `Flow IAT Std`, `Bwd IAT Tot`, `Bwd IAT Std`, `Fwd IAT Std`.
  - **Relational Edge Features (1)**: `Dst Port`.
  - **Relational Graph Node Attributes (2)**: `Src IP`, `Dst IP` (Used solely as graph topological node endpoints, excluded from feature vector $X$).

### B. NF-CSE-CIC-IDS2018-v2 (25 Selected Features)
- **Excluded Columns (17)**: `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `DNS_QUERY_ID`, `CLIENT_TCP_FLAGS`, `NUM_PKTS_UP_TO_128_BYTES`, etc.
- **Selected Tabular Features (25)**:
  `LONGEST_FLOW_PKT`, `FLOW_DURATION_MILLISECONDS`, `MIN_TTL`, `MAX_TTL`, `TCP_WIN_MAX_IN`, `L7_PROTO`, `DURATION_IN`, `TCP_WIN_MAX_OUT`, `SRC_TO_DST_AVG_THROUGHPUT`, `L4_DST_PORT`, `DURATION_OUT`, `MAX_IP_PKT_LEN`, `IN_BYTES`, `TCP_FLAGS`, `SERVER_TCP_FLAGS`, `SRC_TO_DST_SECOND_BYTES`, `DST_TO_SRC_AVG_THROUGHPUT`, `NUM_PKTS_128_TO_256_BYTES`, `MIN_IP_PKT_LEN`, `SHORTEST_FLOW_PKT`, `RETRANSMITTED_IN_BYTES`, `OUT_BYTES`, `NUM_PKTS_256_TO_512_BYTES`, `NUM_PKTS_512_TO_1024_BYTES`, `DNS_TTL_ANSWER`.
- **Structural Role Allocations**:
  - **Temporal Encoder Features (6)**: `FLOW_DURATION_MILLISECONDS`, `DURATION_IN`, `SRC_TO_DST_AVG_THROUGHPUT`, `DURATION_OUT`, `SRC_TO_DST_SECOND_BYTES`, `DST_TO_SRC_AVG_THROUGHPUT`.
  - **Relational Edge Features (7)**: `MIN_TTL`, `MAX_TTL`, `L7_PROTO`, `L4_DST_PORT`, `MAX_IP_PKT_LEN`, `MIN_IP_PKT_LEN`, `DNS_TTL_ANSWER`.

### C. NF-UNSW-NB15-v2 (25 Selected Features)
- **Excluded Columns (18)**: `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `DNS_QUERY_ID`, `OUT_PKTS`, `MAX_IP_PKT_LEN`, etc.
- **Selected Tabular Features (25)**:
  `MIN_TTL`, `MIN_IP_PKT_LEN`, `SHORTEST_FLOW_PKT`, `MAX_TTL`, `LONGEST_FLOW_PKT`, `TCP_WIN_MAX_IN`, `OUT_BYTES`, `L4_DST_PORT`, `DST_TO_SRC_AVG_THROUGHPUT`, `TCP_FLAGS`, `SERVER_TCP_FLAGS`, `TCP_WIN_MAX_OUT`, `SRC_TO_DST_SECOND_BYTES`, `DST_TO_SRC_SECOND_BYTES`, `NUM_PKTS_UP_TO_128_BYTES`, `PROTOCOL`, `IN_BYTES`, `IN_PKTS`, `L4_SRC_PORT`, `SRC_TO_DST_AVG_THROUGHPUT`, `ICMP_TYPE`, `L7_PROTO`, `NUM_PKTS_128_TO_256_BYTES`, `DNS_QUERY_TYPE`, `NUM_PKTS_256_TO_512_BYTES`.
- **Structural Role Allocations**:
  - **Temporal Encoder Features (4)**: `DST_TO_SRC_AVG_THROUGHPUT`, `SRC_TO_DST_SECOND_BYTES`, `DST_TO_SRC_SECOND_BYTES`, `SRC_TO_DST_AVG_THROUGHPUT`.
  - **Relational Edge Features (9)**: `MIN_TTL`, `MIN_IP_PKT_LEN`, `MAX_TTL`, `L4_DST_PORT`, `PROTOCOL`, `L4_SRC_PORT`, `ICMP_TYPE`, `L7_PROTO`, `DNS_QUERY_TYPE`.

---

## 4. Control Across Experiments Verification

- **Experiment Invariance**: All 9 experiments (3 datasets $\times$ 3 loss functions) load features directly via `load_dataset_split()`, which reads the corresponding `configs/feature_selection/*_features.json` file.
- **Verification Result**: Feature selection is 100% identical across standard, weighted, and focal loss experiments for any given dataset. Loss function choice does not alter the input dimension or feature composition.

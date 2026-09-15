# Dataset Metadata: sample_flow

*Generated automatically by Adaptive Temporal-Relational IDS Engine on 2026-09-15 18:57:28*

---

## A. Basic Information
- **dataset_name:** sample_flow
- **dataset_version:** N/A
- **publication_year:** N/A
- **dataset_release_year:** N/A
- **authors/creators:** N/A
- **official_source:** N/A
- **kaggle_source:** N/A
- **role_tier:** Tier Unassigned - N/A
- **description:** N/A

## B. Dataset Type & Lineage
- **dataset_type:** Flow / NetFlow (Format: `csv`)
- **schema_type:** Custom
- **underlying_traffic_source:** Independent PCAP capture

## C. Dataset Size & Physical Metrics
- **file_name:** `sample_flow.csv`
- **file_size_mb:** 0.0 MB (0.0 GB)
- **total_records:** 5
- **total_features:** 14

## D. Critical Features for Temporal-Relational Research
- **Temporal-Relational Compatibility Score:** **90 / 100**
- **Graph Construction Ready ($G_t = (V_t, E_t)$):** `YES`
- **Temporal Ordering Ready:** `YES`
- **NetFlow 5-Tuple Completeness:** `YES`

### Endpoint Identifiers (Graph Nodes & Edges)
- **Source IP:** `IPV4_SRC_ADDR`
- **Destination IP:** `IPV4_DST_ADDR`
- **Source Port:** `L4_SRC_PORT`
- **Destination Port:** `L4_DST_PORT`
- **Protocol:** `PROTOCOL`

## E. Temporal Dynamics & Flow Durations
- **Timestamp Feature:** `FLOW_START_MILLISECONDS`
  - *Sample Value:* `1600000000000`
  - *Is Numeric / Epoch:* `True`
- **Flow Duration Feature:** `FLOW_DURATION_MILLISECONDS`
- **Inter-Arrival Time (IAT) Features:** `None`
- **Active / Idle Time Features:** `None`

## F. Flow Volume & Rate Indicators
- **Packet Count Columns:** `IN_PKTS, OUT_PKTS`
- **Byte Count Columns:** `IN_BYTES, OUT_BYTES`
- **Flow Rate Columns:** `None`

## G. Label & Ground Truth Distribution
### Column: `Label`
| Class / Label | Sample Count |
| :--- | :--- |
| `1` | 3 |
| `0` | 2 |

### Column: `Attack`
| Class / Label | Sample Count |
| :--- | :--- |
| `Benign` | 2 |
| `SSH-Bruteforce` | 2 |
| `DoS-HTTP` | 1 |

## H. Complete Column Schema
| # | Column Name | Data Type | Null % | Sample Values |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `IPV4_SRC_ADDR` | `object` | 0.0% | `192.168.1.100, 192.168.1.101, 192.168...` |
| 2 | `L4_SRC_PORT` | `int64` | 0.0% | `54321, 54322, 48920` |
| 3 | `IPV4_DST_ADDR` | `object` | 0.0% | `10.0.0.1, 10.0.0.2, 10.0.0.1` |
| 4 | `L4_DST_PORT` | `int64` | 0.0% | `80, 443, 22` |
| 5 | `PROTOCOL` | `int64` | 0.0% | `6, 6, 6` |
| 6 | `L4_TCP_FLAGS` | `int64` | 0.0% | `24, 27, 2` |
| 7 | `FLOW_DURATION_MILLISECONDS` | `int64` | 0.0% | `1520, 320, 45` |
| 8 | `IN_BYTES` | `int64` | 0.0% | `4096, 1200, 120` |
| 9 | `OUT_BYTES` | `int64` | 0.0% | `8192, 4500, 60` |
| 10 | `IN_PKTS` | `int64` | 0.0% | `20, 10, 2` |
| 11 | `OUT_PKTS` | `int64` | 0.0% | `35, 15, 1` |
| 12 | `FLOW_START_MILLISECONDS` | `int64` | 0.0% | `1600000000000, 1600000000500, 1600000...` |
| 13 | `Label` | `int64` | 0.0% | `0, 0, 1` |
| 14 | `Attack` | `object` | 0.0% | `Benign, Benign, SSH-Bruteforce` |
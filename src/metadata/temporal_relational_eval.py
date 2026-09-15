"""
Temporal and Relational Feature Evaluator.
Analyzes column names and sample values to assess readiness for Graph and Sequence modeling.
"""

from typing import Dict, List, Any, Optional
import re


def evaluate_temporal_relational_readiness(columns: List[str], sample_df_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Inspects column schema to identify and classify temporal and relational attributes.
    
    Args:
        columns: List of column names in the dataset.
        sample_df_dict: Optional dictionary of sample values per column.
    
    Returns:
        Structured evaluation dict with scores, mapped fields, and readiness flags.
    """
    col_lower_map = {c.lower().strip().replace(" ", "_").replace("-", "_").replace(".", "_"): c for c in columns}
    
    # 1. Relational Feature Discovery (IP, Port, Protocol)
    src_ip_patterns = [r"^ipv4_src_addr$", r"^src_ip$", r"^source_ip$", r"^srcip$", r"^src_address$", r"^ip_src$"]
    dst_ip_patterns = [r"^ipv4_dst_addr$", r"^dst_ip$", r"^destination_ip$", r"^dstip$", r"^dst_address$", r"^ip_dst$"]
    src_port_patterns = [r"^l4_src_port$", r"^src_port$", r"^source_port$", r"^sport$", r"^src_pt$"]
    dst_port_patterns = [r"^l4_dst_port$", r"^dst_port$", r"^destination_port$", r"^dsport$", r"^dst_pt$"]
    proto_patterns = [r"^protocol$", r"^proto$", r"^l4_proto$"]

    def find_match(patterns: List[str]) -> Optional[str]:
        for norm_col, orig_col in col_lower_map.items():
            for pat in patterns:
                if re.search(pat, norm_col):
                    return orig_col
        return None

    src_ip = find_match(src_ip_patterns)
    dst_ip = find_match(dst_ip_patterns)
    src_port = find_match(src_port_patterns)
    dst_port = find_match(dst_port_patterns)
    protocol = find_match(proto_patterns)

    # 2. Temporal Feature Discovery
    timestamp_patterns = [
        r"^timestamp$", r"^flow_start.*$", r"^stime$", r"^start_time$", 
        r"^time$", r"^flow_start_timestamp$", r"^first_switched$", r"^epoch_time$"
    ]
    duration_patterns = [r"^flow_duration.*$", r"^dur$", r"^duration.*$"]
    iat_patterns = [r".*iat.*", r".*inter_arrival.*"]
    active_idle_patterns = [r".*active.*", r".*idle.*"]

    timestamp = find_match(timestamp_patterns)
    duration = find_match(duration_patterns)
    iat_cols = [orig for norm, orig in col_lower_map.items() if any(re.search(p, norm) for p in iat_patterns)]
    active_idle_cols = [orig for norm, orig in col_lower_map.items() if any(re.search(p, norm) for p in active_idle_patterns)]

    # 3. Flow Volume / Statistics
    byte_cols = [orig for norm, orig in col_lower_map.items() if "byte" in norm or "bytes" in norm or "octets" in norm]
    packet_cols = [orig for norm, orig in col_lower_map.items() if "pkt" in norm or "packet" in norm or "pkts" in norm]
    rate_cols = [orig for norm, orig in col_lower_map.items() if "rate" in norm or "/s" in norm or "per_sec" in norm]

    # 4. Label Discovery
    label_patterns = [r"^label$", r"^attack$", r"^attack_cat$", r"^label_multi$", r"^class$", r"^category$"]
    label_cols = [orig for norm, orig in col_lower_map.items() if any(re.search(p, norm) for p in label_patterns)]

    # 5. Readiness Assessment
    graph_ready = bool(src_ip and dst_ip)
    temporal_order_ready = bool(timestamp or duration)
    netflow_standard_ready = bool(src_ip and dst_ip and src_port and dst_port and protocol)

    # Calculate Compatibility Score (0 - 100)
    score = 0
    if graph_ready: score += 25
    if src_port and dst_port: score += 15
    if protocol: score += 10
    if timestamp: score += 20
    if duration: score += 10
    if iat_cols: score += 10
    if label_cols: score += 10

    return {
        "score": score,
        "graph_ready": graph_ready,
        "temporal_order_ready": temporal_order_ready,
        "netflow_standard_ready": netflow_standard_ready,
        "relational_fields": {
            "source_ip": src_ip,
            "destination_ip": dst_ip,
            "source_port": src_port,
            "destination_port": dst_port,
            "protocol": protocol,
        },
        "temporal_fields": {
            "timestamp": timestamp,
            "duration": duration,
            "iat_features": iat_cols,
            "active_idle_features": active_idle_cols,
        },
        "volume_fields": {
            "byte_features": byte_cols[:8],
            "packet_features": packet_cols[:8],
            "rate_features": rate_cols[:8],
        },
        "label_fields": label_cols,
    }

"""
Dynamic Relational Graph Engine.
Constructs directed multi-graph snapshots G_t = (V_t, E_t, X_v, X_e) for temporal windows
using strictly genuine network IP endpoints and features without synthetic fallbacks or dummy nodes.
"""

from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch

try:
    from torch_geometric.data import Data as PyGData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    PyGData = None


class PyGGraphSnapshotEngine:
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols

    def _compute_entropy(self, values: List[Any]) -> float:
        if not values:
            return 0.0
        counts = pd.Series(values).value_counts()
        probs = counts / len(values)
        return float(-np.sum(probs * np.log2(probs + 1e-12)))

    def build_snapshot_from_window(
        self, window_df: pd.DataFrame, binary_label: int = 0, attack_family: str = "BENIGN"
    ) -> Any:
        """
        Builds a single graph snapshot G_t from a DataFrame slice representing window W_t.
        Strictly requires genuine 'src_ip' and 'dst_ip' endpoints and exact feature columns.
        """
        if window_df.empty:
            raise ValueError("Cannot construct graph snapshot from an empty window DataFrame.")

        # Require real IP endpoints - no synthetic node fallbacks
        if "src_ip" not in window_df.columns or "dst_ip" not in window_df.columns:
            raise ValueError(
                "Graph snapshot generation requires real 'src_ip' and 'dst_ip' columns in window_df. "
                "Synthetic IP fallbacks have been strictly removed."
            )

        src_ips = window_df["src_ip"].astype(str).tolist()
        dst_ips = window_df["dst_ip"].astype(str).tolist()

        unique_nodes = list(set(src_ips + dst_ips))
        node2id = {node: i for i, node in enumerate(unique_nodes)}

        src_indices = [node2id[ip] for ip in src_ips]
        dst_indices = [node2id[ip] for ip in dst_ips]

        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)

        # Exact feature matrix - no synthetic zero-padding
        missing_feats = [c for c in self.feature_cols if c not in window_df.columns]
        if missing_feats:
            raise KeyError(f"Window dataframe missing required exact feature columns: {missing_feats}")

        edge_feat_np = window_df[self.feature_cols].to_numpy(dtype=np.float32)
        edge_attr = torch.tensor(edge_feat_np, dtype=torch.float32)

        # Node feature matrix X_v (In-degree, Out-degree, Destination Port Entropy, Volume Ratios)
        num_nodes = len(unique_nodes)
        node_features = np.zeros((num_nodes, 5), dtype=np.float32)

        in_degrees = np.zeros(num_nodes, dtype=np.float32)
        out_degrees = np.zeros(num_nodes, dtype=np.float32)

        for src_i, dst_i in zip(src_indices, dst_indices):
            out_degrees[src_i] += 1.0
            in_degrees[dst_i] += 1.0

        for node_str, node_id in node2id.items():
            node_features[node_id, 0] = in_degrees[node_id]
            node_features[node_id, 1] = out_degrees[node_id]
            
            # Real destination port entropy for source node
            dst_ports = window_df[window_df["src_ip"] == node_str]["dst_port"].tolist() if "dst_port" in window_df.columns else []
            node_features[node_id, 2] = self._compute_entropy(dst_ports)

            # Real Transmitted / Received volume ratio
            fwd_b = window_df[window_df["src_ip"] == node_str]["total_fwd_bytes"].sum() if "total_fwd_bytes" in window_df.columns else 0.0
            bwd_b = window_df[window_df["dst_ip"] == node_str]["total_bwd_bytes"].sum() if "total_bwd_bytes" in window_df.columns else 0.0
            node_features[node_id, 3] = float(fwd_b / (bwd_b + 1e-5))
            node_features[node_id, 4] = float((in_degrees[node_id] + 1) / (out_degrees[node_id] + 1))

        x = torch.tensor(node_features, dtype=torch.float32)
        y = torch.tensor([binary_label], dtype=torch.long)

        data_dict = {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y": y,
            "attack_family": attack_family,
            "num_nodes": num_nodes,
        }

        if HAS_PYG:
            return PyGData(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                attack_family=attack_family,
                num_nodes=num_nodes
            )
        else:
            return data_dict

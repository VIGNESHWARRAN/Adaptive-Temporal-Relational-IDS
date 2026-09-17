"""
Temporal Windowing and Inter-Flow Velocity Engine.
Sorts flow records chronologically and extracts temporal dynamics and slotted sequence windows S_t
using strictly genuine dataset timestamps and labels without synthetic gap filling.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


class TemporalSequenceEngine:
    def __init__(self, sequence_length_k: int = 64, feature_cols: Optional[List[str]] = None):
        self.k = sequence_length_k
        self.feature_cols = feature_cols

    def compute_inter_flow_deltas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes entity-level flow deltas strictly using real dataset ordering and IP fields:
        - delta_t_prev_src
        - delta_t_prev_dst
        - delta_t_pair
        """
        df = df.copy()

        # Sort chronologically if real timestamp exists, otherwise preserve natural capture order
        if "timestamp_start" in df.columns:
            df = df.sort_values("timestamp_start").reset_index(drop=True)
            time_series = df["timestamp_start"]
        else:
            time_series = pd.Series(np.arange(len(df), dtype=np.float64), index=df.index)

        # Require real IP endpoints for entity-level deltas
        if "src_ip" not in df.columns or "dst_ip" not in df.columns:
            raise ValueError("Inter-flow velocity calculation requires real 'src_ip' and 'dst_ip' columns.")

        df["delta_t_prev_src"] = df.groupby("src_ip")[time_series.name if hasattr(time_series, 'name') else "timestamp_start"].diff().fillna(0.0) if "timestamp_start" in df.columns else df.groupby("src_ip").cumcount().astype(np.float32)
        df["delta_t_prev_dst"] = df.groupby("dst_ip")[time_series.name if hasattr(time_series, 'name') else "timestamp_start"].diff().fillna(0.0) if "timestamp_start" in df.columns else df.groupby("dst_ip").cumcount().astype(np.float32)
        df["delta_t_pair"] = df.groupby(["src_ip", "dst_ip"])[time_series.name if hasattr(time_series, 'name') else "timestamp_start"].diff().fillna(0.0) if "timestamp_start" in df.columns else df.groupby(["src_ip", "dst_ip"]).cumcount().astype(np.float32)

        return df

    def extract_sequence_windows(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Slices chronologically ordered flows into event sequence matrices S_t of shape (N_windows, K, d_features).
        Strictly requires exact feature columns without zero-padding missing features.
        """
        df_processed = self.compute_inter_flow_deltas(df)
        
        # Verify exact feature columns exist
        missing_cols = [c for c in feature_cols if c not in df_processed.columns]
        if missing_cols:
            raise KeyError(f"DataFrame missing required sequence feature columns: {missing_cols}")

        feat_matrix = df_processed[feature_cols].to_numpy(dtype=np.float32)

        if "label_binary" not in df_processed.columns or "label_attack_family" not in df_processed.columns:
            raise KeyError("DataFrame missing harmonized label columns 'label_binary' and 'label_attack_family'.")

        binary_labels = df_processed["label_binary"].to_numpy(dtype=np.int64)
        families = df_processed["label_attack_family"].to_list()

        n_samples = len(df_processed)
        if n_samples < self.k:
            raise ValueError(f"Dataset slice contains only {n_samples} flows, which is less than sequence length K={self.k}.")

        n_windows = n_samples // self.k
        sequences = []
        labels = []
        attack_families = []

        for i in range(n_windows):
            start_idx = i * self.k
            end_idx = start_idx + self.k
            seq = feat_matrix[start_idx:end_idx]
            
            win_labels = binary_labels[start_idx:end_idx]
            win_fam = families[start_idx:end_idx]
            
            is_attack = int(np.any(win_labels > 0))
            dominant_fam = "BENIGN"
            if is_attack:
                attack_fams = [f for f in win_fam if f != "BENIGN"]
                if not attack_fams:
                    raise ValueError("Window labeled as attack but no non-BENIGN attack family found in slice.")
                dominant_fam = max(set(attack_fams), key=attack_fams.count)

            sequences.append(seq)
            labels.append(is_attack)
            attack_families.append(dominant_fam)

        return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64), attack_families

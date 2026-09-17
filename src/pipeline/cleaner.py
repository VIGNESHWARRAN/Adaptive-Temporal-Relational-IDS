"""
Data Cleaner and Preprocessor.
Handles inf/NaN values, type casting, memory downcasting, unit scaling,
and exact canonical feature transformations (e.g. converting CICFlowMeter TCP flag counts to bitmask).
"""

import numpy as np
import pandas as pd
from typing import List, Optional


class DataCleaner:
    def __init__(self, memory_downcast: bool = True):
        self.memory_downcast = memory_downcast

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans infinite/NaN values and safely downcasts float64 -> float32.
        All float64 values are clipped to the float32 representable range before
        casting to prevent RuntimeWarning overflows from extreme rate/throughput
        values (e.g. SRC_TO_DST_AVG_THROUGHPUT when flow duration -> 0).
        Integer columns are kept as int64 to prevent overflow of large byte counters.
        """
        df = df.copy()

        _F32_MAX = float(np.finfo(np.float32).max)
        _F32_MIN = float(np.finfo(np.float32).min)

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # Replace inf/-inf with NaN first, then zero-fill
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if df[col].isnull().any():
                df[col] = df[col].fillna(0.0)

        # Safe float64 -> float32 downcast: clip to float32 range first
        # so pandas never encounters an unrepresentable value during astype().
        if self.memory_downcast:
            for col in numeric_cols:
                if df[col].dtype == np.float64:
                    df[col] = df[col].clip(lower=_F32_MIN, upper=_F32_MAX).astype(np.float32)

        return df

    def standardize_units(self, df: pd.DataFrame, dataset_format: str = "netflow_v2") -> pd.DataFrame:
        """
        Ensures consistent units and exact feature transformations.
        Converts microsec -> ms, maps CICFlowMeter flag counts -> tcp_flags bitmask, and rates -> throughput.
        """
        df = df.copy()

        if dataset_format == "cicflowmeter":
            if "flow_duration_ms" in df.columns:
                # CICFlowMeter stores duration in microseconds; convert to ms
                df["flow_duration_ms"] = df["flow_duration_ms"] / 1000.0

            # Derive tcp_flags from individual flag count columns if missing
            if "tcp_flags" not in df.columns:
                fin = df["fin_flag_cnt"] if "fin_flag_cnt" in df.columns else 0
                syn = df["syn_flag_cnt"] if "syn_flag_cnt" in df.columns else 0
                rst = df["rst_flag_cnt"] if "rst_flag_cnt" in df.columns else 0
                psh = df["psh_flag_cnt"] if "psh_flag_cnt" in df.columns else 0
                ack = df["ack_flag_cnt"] if "ack_flag_cnt" in df.columns else 0
                urg = df["urg_flag_cnt"] if "urg_flag_cnt" in df.columns else 0

                df["tcp_flags"] = (fin * 1) + (syn * 2) + (rst * 4) + (psh * 8) + (ack * 16) + (urg * 32)

            # Map CICFlowMeter rates to throughput
            if "fwd_throughput" not in df.columns and "fwd_byte_rate" in df.columns:
                df["fwd_throughput"] = df["fwd_byte_rate"]
            if "bwd_throughput" not in df.columns and "fwd_packet_rate" in df.columns:
                df["bwd_throughput"] = df["fwd_packet_rate"]

        return df

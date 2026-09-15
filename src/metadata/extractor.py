"""
Metadata Extractor Engine.
Extracts schema definitions, data types, null rates, record counts,
class distributions, and temporal-relational indicators from flow datasets.
"""

import os
import json
import gzip
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

from .temporal_relational_eval import evaluate_temporal_relational_readiness


class MetadataExtractor:
    """Memory-efficient metadata extraction and data profiling engine."""

    def __init__(self, sample_rows: int = 20000):
        self.sample_rows = sample_rows

    def _get_file_size(self, file_path: Path) -> Dict[str, Any]:
        """Get file size in Bytes, MB, and GB."""
        size_bytes = file_path.stat().st_size
        return {
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "size_gb": round(size_bytes / (1024 * 1024 * 1024), 3),
        }

    def _count_lines_fast(self, file_path: Path) -> int:
        """Fast newline counter without loading full dataset into memory."""
        try:
            if str(file_path).endswith(".gz"):
                with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
                    return sum(1 for _ in f) - 1 # Minus header
            else:
                with open(file_path, "rb") as f:
                    lines = 0
                    buf_size = 1024 * 1024 * 8
                    read_f = f.raw.read if hasattr(f, "raw") else f.read
                    buf = read_f(buf_size)
                    while buf:
                        lines += buf.count(b"\n")
                        buf = read_f(buf_size)
                    return max(0, lines - 1) # Minus header
        except Exception:
            return -1

    def profile_file(self, file_path: Union[str, Path], dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract detailed metadata profile for a single data file (.csv, .csv.gz, .parquet).
        
        Args:
            file_path: Path to dataset file
            dataset_name: Optional name tag
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if dataset_name is None:
            dataset_name = path.stem.replace(".csv", "").replace(".parquet", "")

        file_meta = self._get_file_size(path)
        print(f"[INFO] Profiling {dataset_name} ({path.name}, {file_meta['size_mb']} MB)...")

        # Read sample slice for rapid profiling
        if str(path).endswith(".parquet"):
            sample_df = pd.read_parquet(path).head(self.sample_rows)
            total_records = len(pd.read_parquet(path, columns=[sample_df.columns[0]]))
        else:
            sample_df = pd.read_csv(path, nrows=self.sample_rows, low_memory=False)
            total_records = self._count_lines_fast(path)
            if total_records <= 0:
                total_records = len(sample_df)

        columns = list(sample_df.columns)
        num_features = len(columns)

        # Inspect data types and null percentages
        col_profiles = []
        for col in columns:
            series = sample_df[col]
            null_pct = float(series.isna().mean() * 100)
            dtype_str = str(series.dtype)
            
            unique_count = int(series.nunique())
            sample_vals = series.dropna().head(3).tolist()
            # Convert non-serializable types to strings
            sample_vals = [str(v) if isinstance(v, (np.generic, pd.Timestamp)) else v for v in sample_vals]

            col_profiles.append({
                "column_name": col,
                "data_type": dtype_str,
                "null_percent": round(null_pct, 2),
                "unique_values_sample": unique_count,
                "sample_values": sample_vals,
            })

        # Evaluate Temporal & Relational readiness
        temp_rel_eval = evaluate_temporal_relational_readiness(columns)

        # Class Distribution Analysis
        labels_distribution = {}
        for lbl_col in temp_rel_eval["label_fields"]:
            if lbl_col in sample_df.columns:
                counts = sample_df[lbl_col].value_counts(dropna=False).head(15).to_dict()
                # Ensure json-serializable keys
                labels_distribution[lbl_col] = {str(k): int(v) for k, v in counts.items()}

        # Timestamp Analysis
        time_info = {}
        ts_col = temp_rel_eval["temporal_fields"]["timestamp"]
        if ts_col and ts_col in sample_df.columns:
            ts_series = sample_df[ts_col].dropna()
            if not ts_series.empty:
                time_info["column"] = ts_col
                time_info["sample_value"] = str(ts_series.iloc[0])
                time_info["is_numeric"] = pd.api.types.is_numeric_dtype(ts_series)
                time_info["min_sample"] = str(ts_series.min())
                time_info["max_sample"] = str(ts_series.max())

        profile = {
            "dataset_name": dataset_name,
            "file_name": path.name,
            "file_path": str(path.resolve()),
            "file_size": file_meta,
            "total_records_estimate": total_records,
            "total_features": num_features,
            "columns": columns,
            "column_profiles": col_profiles,
            "temporal_relational_evaluation": temp_rel_eval,
            "labels_distribution": labels_distribution,
            "temporal_summary": time_info,
        }

        return profile

    def profile_directory(self, dir_path: Union[str, Path]) -> Dict[str, Any]:
        """Profile all CSV, CSV.GZ, and Parquet files in a target directory."""
        directory = Path(dir_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        files = list(directory.glob("*.csv")) + list(directory.glob("*.csv.gz")) + list(directory.glob("*.parquet"))
        
        # Also check immediate subfolders (e.g. data/raw/nf-unsw-nb15-v2/)
        for sub in directory.iterdir():
            if sub.is_dir():
                files.extend(list(sub.glob("*.csv")) + list(sub.glob("*.csv.gz")) + list(sub.glob("*.parquet")))

        profiles = {}
        for f in files:
            ds_name = f.parent.name if f.parent != directory else f.stem
            profiles[f.name] = self.profile_file(f, dataset_name=ds_name)

        return profiles

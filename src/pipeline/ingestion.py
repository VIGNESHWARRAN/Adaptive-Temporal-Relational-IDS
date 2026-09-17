"""
Streaming and Chunked CSV Loader for large network intrusion datasets.
Supports loading datasets exceeding RAM (~19M rows) in memory-efficient chunks.
Includes robust case-insensitive and whitespace-invariant column header alias matching.
"""

import os
import glob
from typing import Generator, List, Union, Optional, Dict
import pandas as pd
import yaml


class ChunkedDatasetLoader:
    def __init__(self, config_path: str = "configs/canonical_schema.yaml"):
        with open(config_path, "r") as f:
            self.schema_config = yaml.safe_load(f)
        self.dataset_mappings = self.schema_config.get("datasets", {})

    def _resolve_files(self, data_path: str) -> List[str]:
        if os.path.isfile(data_path):
            return [data_path]
        elif os.path.isdir(data_path):
            files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
            if not files:
                # search recursively if nested
                files = sorted(glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True))
            return files
        else:
            raise FileNotFoundError(f"Path does not exist: {data_path}")

    def _build_header_map(self, raw_columns: List[str], col_mapping: Dict[str, str]) -> Dict[str, str]:
        """
        Builds a case-insensitive and space-trimmed mapping from actual CSV headers to canonical names.
        """
        norm_col_mapping = {k.strip().lower(): v for k, v in col_mapping.items()}
        mapped_rename = {}

        for orig_col in raw_columns:
            clean_col = orig_col.strip()
            norm_col = clean_col.lower()

            if norm_col in norm_col_mapping:
                mapped_rename[orig_col] = norm_col_mapping[norm_col]

        return mapped_rename

    def load_chunks(
        self,
        dataset_key: str,
        data_path: str,
        chunksize: int = 100_000,
        sample_limit: Optional[int] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Yields pandas DataFrames mapped to canonical column names in chunks.
        """
        if dataset_key not in self.dataset_mappings:
            raise ValueError(f"Unknown dataset_key: '{dataset_key}'. Must be one of {list(self.dataset_mappings.keys())}")

        col_mapping = self.dataset_mappings[dataset_key]["columns"]
        files = self._resolve_files(data_path)

        rows_yielded = 0
        for file_path in files:
            for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
                header_map = self._build_header_map(list(chunk.columns), col_mapping)
                
                # Filter chunk to mapped columns and rename
                chunk_mapped = chunk[list(header_map.keys())].rename(columns=header_map)
                chunk_mapped["provenance_dataset_id"] = dataset_key
                chunk_mapped["source_filename"] = os.path.basename(file_path)

                if sample_limit is not None:
                    needed = sample_limit - rows_yielded
                    if len(chunk_mapped) > needed:
                        chunk_mapped = chunk_mapped.iloc[:needed]
                    rows_yielded += len(chunk_mapped)
                    yield chunk_mapped
                    if rows_yielded >= sample_limit:
                        return
                else:
                    yield chunk_mapped

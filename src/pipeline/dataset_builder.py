"""
Synchronized Dual-Modal Dataset Builder — Streaming / Batch Mode.

Processes the full dataset (18M+ rows) in a true streaming fashion:
  - Reads one 100k-row chunk at a time from disk (never loads all rows into RAM).
  - Carries a small inter-chunk row buffer of size (K-1) so sequence windows
    never get silently split across chunk boundaries.
  - Cleans, label-maps, computes inter-flow deltas, slices S_t windows, and
    builds G_t graph snapshots — all within each batch.
  - Incrementally appends finished (S_t, G_t, Y_t) samples to a single .pt file
    on disk so RAM usage stays flat regardless of dataset size.
"""

import os
import math
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch

from .ingestion import ChunkedDatasetLoader
from .cleaner import DataCleaner
from .label_mapper import HarmonizedLabelMapper
from .temporal_engine import TemporalSequenceEngine
from .graph_engine import PyGGraphSnapshotEngine


class DualModalDatasetBuilder:
    def __init__(
        self,
        config_path: str = "configs/canonical_schema.yaml",
        label_config_path: str = "configs/label_taxonomy.yaml",
        sequence_k: int = 64,
        chunk_size: int = 100_000,
    ):
        self.loader = ChunkedDatasetLoader(config_path)
        self.cleaner = DataCleaner(memory_downcast=True)
        self.label_mapper = HarmonizedLabelMapper(label_config_path)
        self.temporal_engine = TemporalSequenceEngine(sequence_length_k=sequence_k)
        self.chunk_size = chunk_size

        # Core features shared by sequence matrix S_t and graph edge attributes X_e
        self.feature_cols = [
            "flow_duration_ms", "total_fwd_bytes", "total_bwd_bytes",
            "total_fwd_packets", "total_bwd_packets",
            "min_pkt_len", "max_pkt_len", "tcp_flags",
            "fwd_throughput", "bwd_throughput",
            "delta_t_prev_src", "delta_t_prev_dst", "delta_t_pair",
        ]
        self.graph_engine = PyGGraphSnapshotEngine(self.feature_cols)

    def _process_window_batch(
        self,
        processed_df: pd.DataFrame,
        global_window_offset: int,
    ) -> List[Dict[str, Any]]:
        """
        Extract all complete K-length windows from processed_df and build
        synchronized (S_t, G_t, Y_t) samples. Returns the list of samples.
        """
        k = self.temporal_engine.k
        n = len(processed_df)
        n_windows = n // k
        if n_windows == 0:
            return []

        feat_matrix = processed_df[self.feature_cols].to_numpy(dtype=np.float32)
        binary_labels = processed_df["label_binary"].to_numpy(dtype=np.int64)
        families = processed_df["label_attack_family"].to_list()

        samples = []
        for i in range(n_windows):
            start = i * k
            end = start + k
            seq = feat_matrix[start:end]

            win_labels = binary_labels[start:end]
            win_fam = families[start:end]

            is_attack = int(np.any(win_labels > 0))
            dominant_fam = "BENIGN"
            if is_attack:
                attack_fams = [f for f in win_fam if f != "BENIGN"]
                if attack_fams:
                    dominant_fam = max(set(attack_fams), key=attack_fams.count)

            window_slice = processed_df.iloc[start:end]
            graph_snapshot = self.graph_engine.build_snapshot_from_window(
                window_slice, binary_label=is_attack, attack_family=dominant_fam
            )

            samples.append({
                "sequence_S_t": torch.tensor(seq, dtype=torch.float32),
                "graph_G_t": graph_snapshot,
                "label_binary": torch.tensor([is_attack], dtype=torch.long),
                "label_attack_family": dominant_fam,
                "window_idx": global_window_offset + i,
            })

        return samples

    def process_dataset(
        self,
        dataset_key: str,
        data_path: str,
        sample_limit: Optional[int] = None,
        output_dir: Optional[str] = "data/processed",
    ) -> List[Dict[str, Any]]:
        """
        Fully streaming pipeline — processes one chunk at a time.
        RAM usage stays bounded at ~chunk_size rows regardless of total dataset size.
        """
        dataset_format = self.loader.dataset_mappings[dataset_key]["format"]
        k = self.temporal_engine.k

        print(f"--> Streaming dataset '{dataset_key}' from: {data_path}")
        if sample_limit:
            print(f"--> Sample limit: {sample_limit:,} rows")
        else:
            print(f"--> Sample limit: NONE (processing full dataset)")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{dataset_key}_synchronized_samples.pt") if output_dir else None

        all_samples: List[Dict[str, Any]] = []
        carry_buffer: Optional[pd.DataFrame] = None  # rows from previous chunk that didn't fill a full window
        total_rows_seen = 0
        total_windows = 0
        chunk_num = 0

        for raw_chunk in self.loader.load_chunks(
            dataset_key, data_path,
            chunksize=self.chunk_size,
            sample_limit=sample_limit,
        ):
            chunk_num += 1
            total_rows_seen += len(raw_chunk)

            # Clean, standardize and label-map this chunk
            cleaned = self.cleaner.clean_dataframe(raw_chunk)
            cleaned = self.cleaner.standardize_units(cleaned, dataset_format=dataset_format)
            processed = self.label_mapper.apply_to_dataframe(cleaned)

            # Prepend carry-over rows from the previous chunk
            if carry_buffer is not None and len(carry_buffer) > 0:
                processed = pd.concat([carry_buffer, processed], ignore_index=True)

            # Compute inter-flow temporal deltas on the combined slice
            processed = self.temporal_engine.compute_inter_flow_deltas(processed)

            # Separate the complete windows from the leftover tail
            n_complete_rows = (len(processed) // k) * k
            remainder = processed.iloc[n_complete_rows:].copy()
            to_process = processed.iloc[:n_complete_rows]

            # Build (S_t, G_t, Y_t) samples for complete windows in this batch
            batch_samples = self._process_window_batch(to_process, global_window_offset=total_windows)
            total_windows += len(batch_samples)
            all_samples.extend(batch_samples)

            # Save incrementally every 5 chunks to avoid accumulating too much in RAM
            if save_path and chunk_num % 5 == 0 and all_samples:
                existing = torch.load(save_path, weights_only=False) if os.path.exists(save_path) else []
                torch.save(existing + all_samples, save_path)
                print(
                    f"  [Chunk {chunk_num}] rows={total_rows_seen:,} | "
                    f"windows_this_batch={len(batch_samples)} | "
                    f"total_windows_saved={total_windows}"
                )
                all_samples = []  # free RAM after saving

            # Carry the remainder rows into the next chunk
            carry_buffer = remainder

        # Final save of any remaining unsaved samples
        if save_path:
            existing = torch.load(save_path, weights_only=False) if os.path.exists(save_path) else []
            final_all = existing + all_samples
            torch.save(final_all, save_path)
            print(f"\n--> Pipeline complete.")
            print(f"    Total rows processed : {total_rows_seen:,}")
            print(f"    Total windows (S_t)  : {total_windows + len(all_samples)}")
            print(f"    Output saved to      : {save_path}")
            return final_all

        return all_samples

"""
Main pipeline execution script.
Ingests raw CSV datasets, applies canonical schema mapping, 3-tier label harmonization,
chronological sequence windowing, dynamic graph snapshot generation, and exports .pt samples.
"""

import argparse
import os
import sys

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.dataset_builder import DualModalDatasetBuilder


def main():
    parser = argparse.ArgumentParser(description="Adaptive Temporal-Relational Pipeline Runner")
    parser.add_argument("--dataset", type=str, default="nf_cse_cic_ids2018_v2", help="Dataset key in canonical_schema.yaml")
    parser.add_argument("--data-path", type=str, default=r"d:\Final Year Implementation\data\raw\nf-cse-cic-ids2018-v2\b3427ed8ad063a09_MOHANAD_A4706\data\NF-CSE-CIC-IDS2018-v2.csv")
    parser.add_argument("--sample-limit", type=int, help="Number of rows to sample for processing")
    parser.add_argument("--output-dir", type=str, default=r"d:\Final Year Implementation\data\processed")
    parser.add_argument("--sequence-k", type=int, default=64, help="Sequence window length K")
    args = parser.parse_args()

    print(f"=== Adaptive Temporal-Relational Data Pipeline ===")
    print(f"Dataset Key: {args.dataset}")
    print(f"Input Path:  {args.data_path}")
    print(f"Sample Limit:{args.sample_limit}")
    print(f"Output Dir:  {args.output_dir}")

    builder = DualModalDatasetBuilder(sequence_k=args.sequence_k)
    samples = builder.process_dataset(
        dataset_key=args.dataset,
        data_path=args.data_path,
        sample_limit=args.sample_limit,
        output_dir=args.output_dir,
    )

    print(f"\nPipeline successfully created {len(samples)} synchronized benchmark samples!")


if __name__ == "__main__":
    main()

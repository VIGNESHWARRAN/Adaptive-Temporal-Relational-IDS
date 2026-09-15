#!/usr/bin/env python3
"""
CLI Script: Dataset Metadata Extractor & Profiler
Usage:
    python scripts/extract_metadata.py
    python scripts/extract_metadata.py --data-dir data/raw --output-dir metadata_reports
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.download.dataset_registry import DatasetRegistry
from src.metadata.extractor import MetadataExtractor
from src.metadata.reporter import MetadataReporter
from src.utils.colab_utils import is_colab, setup_colab_environment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect downloaded datasets and generate comprehensive research metadata reports."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to folder containing downloaded raw datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to folder where Markdown and JSON metadata reports will be saved.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=20000,
        help="Number of rows to sample for fast schema profiling (default: 20000).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    registry = DatasetRegistry()

    # Determine environment and paths
    if is_colab():
        env_paths = setup_colab_environment()
        data_dir = Path(args.data_dir) if args.data_dir else env_paths["raw_dir"]
        output_dir = Path(args.output_dir) if args.output_dir else env_paths["metadata_dir"]
    else:
        data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "raw"
        output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "metadata_reports"

    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f" Adaptive Temporal-Relational IDS: Metadata Extraction")
    print(f" Scanning Raw Data Directory: {data_dir}")
    print(f" Saving Metadata Reports To: {output_dir}")
    print(f"=======================================================\n")

    extractor = MetadataExtractor(sample_rows=args.sample_rows)
    reporter = MetadataReporter(output_dir=str(output_dir))

    # Discover datasets in data_dir
    files_to_profile = []
    for pattern in ["*.csv", "*.csv.gz", "*.parquet"]:
        files_to_profile.extend(list(data_dir.glob(pattern)))
        for sub in data_dir.iterdir():
            if sub.is_dir():
                files_to_profile.extend(list(sub.glob(pattern)))

    if not files_to_profile:
        print(f"[WARN] No CSV or Parquet data files found in {data_dir}.")
        print("[HINT] Run `python scripts/download_datasets.py --tier 1` first to download datasets.")
        return

    print(f"[INFO] Found {len(files_to_profile)} data files to profile.\n")

    summary_records = []
    for file_path in files_to_profile:
        try:
            # Map file to dataset registry metadata if possible
            parent_name = file_path.parent.name.lower()
            stem_name = file_path.stem.replace(".csv", "").lower()
            
            matched_id = None
            if parent_name in registry.datasets:
                matched_id = parent_name
            elif stem_name in registry.datasets:
                matched_id = stem_name
            else:
                for k in registry.datasets.keys():
                    if k in str(file_path).lower():
                        matched_id = k
                        break

            reg_meta = registry.get_dataset(matched_id) if matched_id else {}
            display_name = reg_meta.get("name", file_path.stem)

            # Profile file
            profile = extractor.profile_file(file_path, dataset_name=display_name)
            
            # Export JSON and Markdown reports
            reporter.export_json(profile)
            reporter.export_markdown(profile, registry_meta=reg_meta)

            tr_score = profile.get("temporal_relational_evaluation", {}).get("score", 0)
            summary_records.append({
                "name": display_name,
                "records": profile.get("total_records_estimate", 0),
                "features": profile.get("total_features", 0),
                "tr_score": tr_score,
                "size_mb": profile.get("file_size", {}).get("size_mb", 0),
            })
            print()

        except Exception as e:
            print(f"[ERROR] Profiling failed for {file_path.name}: {e}\n")

    print("\n=======================================================")
    print(" METADATA EXTRACTION SUMMARY")
    print("=======================================================")
    for rec in summary_records:
        print(f"[*] {rec['name']}: {rec['records']:,} rows | {rec['features']} feats | TR-Score: {rec['tr_score']}/100 | {rec['size_mb']} MB")
    print(f"\n[DONE] All metadata reports generated in: {output_dir}\n")


if __name__ == "__main__":
    main()

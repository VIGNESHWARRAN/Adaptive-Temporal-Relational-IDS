#!/usr/bin/env python3
"""
CLI Script: Dataset Downloader
Usage:
    python scripts/download_datasets.py --tier 1
    python scripts/download_datasets.py --datasets nf-unsw-nb15-v2 nf-cse-cic-ids2018-v2
    python scripts/download_datasets.py --datasets all --dry-run
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.download.dataset_registry import DatasetRegistry
from src.download.downloader import DatasetDownloader
from src.utils.colab_utils import is_colab, setup_colab_environment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and ingest candidate network IDS datasets for the Temporal-Relational project."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="List of dataset IDs to download (e.g., 'nf-unsw-nb15-v2', 'nf-cse-cic-ids2018-v2', 'all').",
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Filter downloads by research priority tier (1 = Core Primary, 2 = Cross-Domain, 3 = Reference).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target directory for downloaded raw datasets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate and print download plan without fetching actual files.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip automatic decompression of archives.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    registry = DatasetRegistry()

    # Determine environment and paths
    if is_colab():
        env_paths = setup_colab_environment()
        output_dir = Path(args.output_dir) if args.output_dir else env_paths["raw_dir"]
    else:
        output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "raw"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which datasets to download
    target_ids = []
    if args.datasets:
        if "all" in [d.lower() for d in args.datasets]:
            target_ids = registry.get_all_ids()
        else:
            target_ids = [d.lower() for d in args.datasets]
    elif args.tier is not None:
        target_ids = [d["id"] for d in registry.list_datasets(tier=args.tier)]
    else:
        # Default to Tier 1 core primary datasets
        print("[INFO] No dataset or tier specified. Defaulting to Tier 1 Core Primary datasets.")
        target_ids = registry.get_tier1_ids()

    print(f"\n=======================================================")
    print(f" Adaptive Temporal-Relational IDS: Dataset Ingestion")
    print(f" Target Datasets: {target_ids}")
    print(f" Target Directory: {output_dir}")
    print(f" Mode: {'DRY RUN' if args.dry_run else 'ACTIVE DOWNLOAD'}")
    print(f"=======================================================\n")

    downloader = DatasetDownloader(output_dir=str(output_dir))

    for ds_id in target_ids:
        try:
            meta = registry.get_dataset(ds_id)
            meta["id"] = ds_id
            print(f"[*] Dataset: {meta.get('name')} | Tier {meta.get('tier')} | Est. Size: {meta.get('estimated_size_mb')} MB")
            
            if args.dry_run:
                print(f"    - Method: {meta.get('download_method')}")
                if meta.get('kaggle_slug'):
                    print(f"    - Kaggle: {meta.get('kaggle_slug')}")
                if meta.get('direct_url'):
                    print(f"    - Direct URL: {meta.get('direct_url')}")
                if meta.get('mirror_url'):
                    print(f"    - Mirror URL: {meta.get('mirror_url')}")
                print("    - Status: READY")
                continue

            downloader.download_dataset(meta, auto_extract=not args.no_extract)

        except Exception as e:
            print(f"[ERROR] Failed processing {ds_id}: {e}\n")


if __name__ == "__main__":
    main()

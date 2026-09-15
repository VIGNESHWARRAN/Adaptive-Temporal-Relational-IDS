#!/usr/bin/env python3
"""
Master Orchestrator Pipeline Script.
Runs system environment checks, dataset downloads, and metadata extraction in sequence.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.colab_utils import get_system_specs, is_colab
from scripts.download_datasets import main as run_download
from scripts.extract_metadata import main as run_metadata


def print_banner():
    specs = get_system_specs()
    print("=" * 70)
    print(" ADAPTIVE TEMPORAL-RELATIONAL INTRUSION DETECTION PIPELINE")
    print("=" * 70)
    print(f" OS: {specs['os']} ({specs['os_release']}) | Python: {specs['python_version']}")
    print(f" Environment: {'Google Colab' if specs['is_colab'] else 'Local Machine'}")
    if specs.get('total_ram_gb'):
        print(f" System RAM: {specs['total_ram_gb']} GB")
    if specs.get('free_disk_gb'):
        print(f" Free Disk Space: {specs['free_disk_gb']} GB")
    if specs['gpu_available']:
        print(f" GPU: {specs['gpu_name']} (CUDA Enabled)")
    else:
        print(" GPU: None (CPU Mode)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print_banner()
    print("[PHASE 1] Starting Dataset Downloads...")
    run_download()
    
    print("\n[PHASE 2] Starting Metadata Extraction & Profiling...")
    run_metadata()
    
    print("\n[SUCCESS] Pipeline completed successfully!")

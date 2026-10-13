"""Adaptive Temporal-Relational Intrusion Detection System (IDS) - Web UI Launcher.

File-Based Triggering System CLI & Neural SOC Server.

Usage Examples:
    python run_ui.py                 # Default batch size n = 1
    python run_ui.py -n 5            # Batch size n = 5
    python run_ui.py --n 10          # Batch size n = 10
    python run_ui.py 20              # Positional argument n = 20
    python run_ui.py -n 5 --port 8080 --dataset nf_cse_cic_ids2018_v2
"""

import os
import sys
import argparse
import webbrowser
import threading
import time

# Ensure project root is in sys.path
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import uvicorn
from src.ui.engine import IDSEngine
import src.ui.app as ui_app


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive Temporal-Relational IDS - File-Based Triggering Web UI",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Support optional positional batch size or flagged -n / --n
    parser.add_argument(
        "batch_size_pos",
        nargs="?",
        type=int,
        default=None,
        help="Optional positional batch size 'n' (e.g. python run_ui.py 5)",
    )
    parser.add_argument(
        "-n",
        "--n",
        "--batch-size",
        dest="batch_size_flag",
        type=int,
        default=None,
        help="Number of records 'n' to trigger per batch from the CSV file (default: 1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the Web SOC Dashboard on (default: 8050)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--watch-dir",
        type=str,
        default="incoming_flows",
        help="Folder containing incoming network flow CSV files (default: incoming_flows)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nf_cse_cic_ids2018_v2",
        choices=["nf_cse_cic_ids2018_v2", "cse_cic_ids2018", "nf_unsw_nb15_v2"],
        help="Dataset model checkpoint to load (default: nf_cse_cic_ids2018_v2)",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default="concat",
        choices=[
            "concat",
            "gated",
            "bilinear",
            "cross_attention",
            "decision_ensemble",
            "quantum_pqc",
        ],
        help="Multimodal fusion architecture (default: concat)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the browser on startup",
    )

    return parser.parse_args()


def open_browser(url: str):
    """Waits 1.2s for server initialization then opens browser."""
    time.sleep(1.2)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    args = parse_args()

    # Determine batch size 'n' (Flag takes priority, then positional, then default 1)
    if args.batch_size_flag is not None:
        batch_n = max(1, args.batch_size_flag)
    elif args.batch_size_pos is not None:
        batch_n = max(1, args.batch_size_pos)
    else:
        batch_n = 1

    watch_path = os.path.join(REPO_ROOT, args.watch_dir)
    os.makedirs(watch_path, exist_ok=True)

    # Ensure stdout handles UTF-8 on Windows
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    # Initialize Engine
    print("=" * 72)
    print("  [ADAPTIVE-IDS] MULTIMODAL TEMPORAL-RELATIONAL INTRUSION DETECTION")
    print("  [NEURAL-SOC]   FILE-BASED TRIGGERING DASHBOARD")
    print("=" * 72)
    print(f"  * Batch Size Argument (n):    {batch_n} record(s) per trigger")
    print(f"  * Flow Watch Folder:          {watch_path}")
    print(f"  * Active Detection Model:     {args.dataset} (Fusion: {args.fusion})")
    print(f"  * Web Dashboard URL:          http://{args.host}:{args.port}")
    print("=" * 72)

    engine_instance = IDSEngine(
        watch_dir=watch_path,
        default_batch_size=batch_n,
        dataset_name=args.dataset,
        fusion_type=args.fusion,
    )

    # Bind engine to FastAPI app
    ui_app.engine = engine_instance

    url = f"http://{args.host}:{args.port}"

    if not args.no_browser:
        t = threading.Thread(target=open_browser, args=(url,), daemon=True)
        t.start()

    print(f"\n[INFO] Starting Web Server on {url} ... Press CTRL+C to stop.\n")

    uvicorn.run(
        ui_app.app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

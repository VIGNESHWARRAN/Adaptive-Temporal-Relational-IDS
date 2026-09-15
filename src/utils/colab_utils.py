"""
Google Colab Integration & System Diagnostics Utilities.
Provides automated Google Drive mounting, environment detection, and path management.
"""

import os
import sys
import platform
import shutil
from pathlib import Path


def is_colab() -> bool:
    """Check if the current runtime environment is Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def get_system_specs() -> dict:
    """Retrieve host system information (CPU, RAM, GPU, OS)."""
    specs = {
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": sys.version.split()[0],
        "is_colab": is_colab(),
        "gpu_available": False,
        "gpu_name": None,
        "total_ram_gb": None,
        "free_disk_gb": None,
    }

    # Disk space
    try:
        total, used, free = shutil.disk_usage("/")
        specs["free_disk_gb"] = round(free / (1024**3), 2)
    except Exception:
        pass

    # RAM and GPU
    try:
        import psutil
        specs["total_ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            specs["gpu_available"] = True
            specs["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return specs


def setup_colab_environment(drive_project_dir: str = "IDS_Research_Project") -> dict:
    """
    Mount Google Drive and create standardized project directories.
    
    Args:
        drive_project_dir: Folder name in MyDrive where project artifacts will be persisted.
        
    Returns:
        dict with resolved paths for raw data and metadata outputs.
    """
    if not is_colab():
        print("[INFO] Not running in Google Colab. Using local workspace paths.")
        base_dir = Path.cwd()
        raw_dir = base_dir / "data" / "raw"
        metadata_dir = base_dir / "metadata_reports"
    else:
        print("[INFO] Google Colab environment detected. Mounting Google Drive...")
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        
        base_dir = Path("/content/drive/MyDrive") / drive_project_dir
        raw_dir = base_dir / "data" / "raw"
        metadata_dir = base_dir / "metadata_reports"
        
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Raw Data Directory: {raw_dir}")
    print(f"[INFO] Metadata Directory: {metadata_dir}")
    
    return {
        "project_root": base_dir,
        "raw_dir": raw_dir,
        "metadata_dir": metadata_dir,
    }

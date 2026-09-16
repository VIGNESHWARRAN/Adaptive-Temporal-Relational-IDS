"""
Dataset Downloader Engine.
Supports streaming HTTP/HTTPS with resume, KaggleHub/Kaggle CLI, Zenodo endpoints,
and automated archive decompression.
"""

import os
import re
import sys
import gzip
import shutil
import zipfile
import tarfile
import urllib.parse
from pathlib import Path
from typing import Optional, Dict, Any, List
import requests
from tqdm import tqdm


class DatasetDownloader:
    """Downloader supporting multiple protocols and source platforms."""

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_file_http(self, url: str, destination: Path, chunk_size: int = 1024 * 1024) -> Path:
        """
        Download a file over HTTP/HTTPS with progress bar and resume support.
        
        Args:
            url: Remote URL
            destination: Target local file path
            chunk_size: Byte chunk size for streaming (default 1MB)
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_dest = destination.with_suffix(destination.suffix + ".part")
        
        headers = {}
        resume_byte_pos = 0
        if temp_dest.exists():
            resume_byte_pos = temp_dest.stat().st_size
            headers["Range"] = f"bytes={resume_byte_pos}-"
            print(f"[INFO] Resuming download from byte {resume_byte_pos}...")

        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            
            # Check for Content-Disposition filename header to get original archive name
            content_disp = response.headers.get("Content-Disposition", "")
            if content_disp:
                match = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^";\r\n]+)["\']?', content_disp, re.IGNORECASE)
                if match:
                    header_filename = urllib.parse.unquote(match.group(1).strip())
                    if header_filename:
                        destination = destination.parent / header_filename

            # Check for range header support or fresh start
            if response.status_code == 416: # Range not satisfiable (completed)
                if temp_dest.exists():
                    temp_dest.rename(destination)
                    return destination
            elif response.status_code not in (200, 206):
                # Fallback without range headers if server returns 403/500 on range
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                resume_byte_pos = 0

            total_size = int(response.headers.get("content-length", 0)) + resume_byte_pos
            mode = "ab" if resume_byte_pos > 0 else "wb"

            with open(temp_dest, mode) as f, tqdm(
                desc=destination.name,
                initial=resume_byte_pos,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                ncols=90,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Move part file to final destination
            if temp_dest.exists():
                if destination.exists():
                    destination.unlink()
                temp_dest.rename(destination)
                
            print(f"[SUCCESS] Downloaded: {destination}")
            return destination

        except Exception as e:
            print(f"[ERROR] Failed downloading {url}: {e}")
            raise e

    def download_kaggle(self, kaggle_slug: str, target_dir: Path) -> Path:
        """
        Download a dataset from Kaggle using kagglehub or kaggle API.
        
        Args:
            kaggle_slug: Kaggle dataset identifier (e.g. 'mfaizanur/nf-csecicids2018-v2')
            target_dir: Destination folder
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Downloading Kaggle dataset '{kaggle_slug}' into {target_dir}...")
        
        # Try kagglehub first
        try:
            import kagglehub
            path = kagglehub.dataset_download(kaggle_slug)
            print(f"[INFO] Downloaded via KaggleHub to cache: {path}")
            
            # Copy contents to target_dir
            source_path = Path(path)
            if source_path.is_dir():
                for item in source_path.glob("*"):
                    dest_item = target_dir / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest_item, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dest_item)
            else:
                shutil.copy2(source_path, target_dir / source_path.name)
            print(f"[SUCCESS] Kaggle dataset saved to: {target_dir}")
            return target_dir
        except Exception as e1:
            print(f"[WARN] KaggleHub failed ({e1}). Trying Kaggle CLI / API...")

        # Fallback to official Kaggle API
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(kaggle_slug, path=str(target_dir), unzip=True)
            print(f"[SUCCESS] Kaggle API downloaded and extracted to: {target_dir}")
            return target_dir
        except Exception as e2:
            print(f"[ERROR] Kaggle API failed: {e2}")
            print("[HINT] For Kaggle downloads, ensure `kaggle.json` is configured in ~/.kaggle/ or KAGGLE_USERNAME and KAGGLE_KEY environment variables are set.")
            raise e2

    def extract_archive(self, file_path: Path, extract_to: Optional[Path] = None) -> Path:
        """Extract .zip, .tar.gz, .tgz, or .gz files."""
        if extract_to is None:
            extract_to = file_path.parent

        extract_to.mkdir(parents=True, exist_ok=True)
        name_str = str(file_path).lower()

        if name_str.endswith(".zip"):
            print(f"[INFO] Unzipping {file_path.name}...")
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"[SUCCESS] Extracted to {extract_to}")

        elif name_str.endswith((".tar.gz", ".tgz", ".tar")):
            print(f"[INFO] Extracting tar archive {file_path.name}...")
            with tarfile.open(file_path, "r:*") as tar_ref:
                tar_ref.extractall(extract_to)
            print(f"[SUCCESS] Extracted to {extract_to}")

        elif name_str.endswith(".gz") and not name_str.endswith(".tar.gz"):
            decompressed_name = file_path.stem
            out_file = extract_to / decompressed_name
            print(f"[INFO] Decompressing gzip {file_path.name} -> {out_file.name}...")
            with gzip.open(file_path, "rb") as f_in, open(out_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            print(f"[SUCCESS] Decompressed: {out_file}")

        return extract_to

    def download_dataset(self, dataset_meta: Dict[str, Any], auto_extract: bool = True) -> Path:
        """
        Download a dataset according to its metadata configuration.
        
        Args:
            dataset_meta: Dataset dict containing download_method, URLs, kaggle_slug, etc.
            auto_extract: Whether to automatically decompress downloaded archives.
        """
        ds_id = dataset_meta.get("id") or dataset_meta.get("name", "dataset").lower()
        target_dir = self.output_dir / ds_id
        target_dir.mkdir(parents=True, exist_ok=True)

        method = dataset_meta.get("download_method", "direct")
        print(f"\n=======================================================")
        print(f" Downloading: {dataset_meta.get('name')} (Tier {dataset_meta.get('tier')})")
        print(f" Target Directory: {target_dir}")
        print(f" Method: {method}")
        print(f"=======================================================")

        # Method 1: Kaggle
        if method == "kaggle" and dataset_meta.get("kaggle_slug"):
            try:
                return self.download_kaggle(dataset_meta["kaggle_slug"], target_dir)
            except Exception as e:
                print(f"[WARN] Kaggle download encountered an issue. Checking for mirror/direct URL fallback...")
                if not dataset_meta.get("direct_url") and not dataset_meta.get("mirror_url"):
                    raise e

        # Method 2: Direct HTTP / Zenodo / Mirror
        urls_to_try = []
        if dataset_meta.get("direct_url"):
            urls_to_try.append(dataset_meta["direct_url"])
        if dataset_meta.get("mirror_url"):
            urls_to_try.append(dataset_meta["mirror_url"])

        last_err = None
        for url in urls_to_try:
            try:
                # Infer filename from URL
                filename = url.split("?")[0].split("/")[-1]
                if not filename or len(filename) < 3:
                    filename = f"{ds_id}.download"
                dest_file = target_dir / filename
                
                downloaded_file = self.download_file_http(url, dest_file)
                if auto_extract and (
                    downloaded_file.name.lower().endswith((".gz", ".zip", ".tar", ".tgz", ".tar.gz"))
                    or downloaded_file.suffix.lower() in (".gz", ".zip", ".tar", ".tgz")
                ):
                    self.extract_archive(downloaded_file, target_dir)
                return target_dir
            except Exception as e:
                last_err = e
                print(f"[WARN] Failed fetching from {url}: {e}")
                continue

        if last_err:
            raise last_err
        raise RuntimeError(f"No valid download URL or method found for dataset: {dataset_meta.get('name')}")

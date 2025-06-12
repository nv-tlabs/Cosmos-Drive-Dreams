#!/usr/bin/env python3
"""
download_cosmos_drive_dreams.py
--------------------------------
Parallel downloader for
  nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams

• Logical file-types:  hdmap · lidar · synthetic   (any combination)
• Always download common folders:
      all_object_info, captions, car_mask_coarse, ftheta_intrinsic,
      pinhole_intrinsic, pose, vehicle_pose
• Extra folders per type:
      hdmap     → 3d_crosswalks … 3d_wait_lines
      lidar     → lidar_raw
      synthetic → cosmos_synthetic
• Uses a ThreadPoolExecutor (default 8 workers) because downloads are I/O-bound.
• No subset / resolution / hash concept.
"""

from __future__ import annotations
import os, argparse, shutil, traceback, time
from os.path import join
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from huggingface_hub import HfApi, HfFileSystem

# ----------------------------------------------------------------------
DATASET_REPO = "nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams"
COMMON_FOLDERS = [
    "all_object_info",
    "captions",
    "car_mask_coarse",
    "ftheta_intrinsic",
    "pinhole_intrinsic",
    "pose",
    "vehicle_pose",
]
EXTRA_FOLDERS = {
    "hdmap": [
        "3d_crosswalks",
        "3d_lanelines",
        "3d_lanes",
        "3d_poles",
        "3d_road_boundaries",
        "3d_road_markings",
        "3d_traffic_lights",
        "3d_traffic_signs",
        "3d_wait_lines",
    ],
    "lidar": ["lidar_raw"],
    "synthetic": ["cosmos_synthetic"],
}
# ----------------------------------------------------------------------

api = HfApi()
fs = HfFileSystem()


def verify_access(repo: str) -> bool:
    try:
        fs.ls(f"datasets/{repo}")
        return True
    except Exception:
        return False


def list_files(repo: str, folders: list[str]) -> list[str]:
    """Return repo-relative paths of every file under each folder."""
    rel_paths: list[str] = []
    prefix = f"datasets/{repo}/"
    for folder in folders:
        for path in fs.find(f"{prefix}{folder}"):
            rel_paths.append(path[len(prefix):])
    return rel_paths


def hf_download_with_retry(rel: str, odir: str, cache: str, max_try: int = 5) -> bool:
    """Thin retry wrapper around hf_hub_download (network-fragile)."""
    for _ in range(max_try):
        try:
            api.hf_hub_download(
                repo_id=DATASET_REPO,
                filename=rel,
                repo_type="dataset",
                local_dir=odir,
                cache_dir=cache,
                local_dir_use_symlinks=False,
            )
            return True
        except KeyboardInterrupt:
            raise
        except Exception:
            traceback.print_exc()
            time.sleep(1)
    return False


def download_files(rel_paths: list[str], odir: str, clean_cache: bool, num_workers: int) -> bool:
    cache_dir = join(odir, ".hf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    succ = 0

    def download_one(rel: str) -> tuple[str, bool]:
        local_target = join(odir, rel)
        if os.path.exists(local_target):
            return rel, True
        ok = hf_download_with_retry(rel, odir, cache_dir)
        return rel, ok

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(download_one, rel): rel for rel in rel_paths}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading", unit="file"):
            rel, ok = f.result()
            succ += ok
            if not ok:
                print(f"❌ failed: {rel}")

    if clean_cache and os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)

    print(f"✔  {succ}/{len(rel_paths)} files downloaded successfully")
    return succ == len(rel_paths)


def parse_args():
    p = argparse.ArgumentParser("Cosmos-Drive-Dreams downloader (parallel)")
    p.add_argument("--odir", required=True, help="Output directory")
    p.add_argument(
        "--file_types",
        default="hdmap,lidar,synthetic",
        help="Comma-separated subset of {hdmap,lidar,synthetic} (default all)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel download threads (default 1)",
    )
    p.add_argument("--clean_cache", action="store_true", help="Delete HF cache after download")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.odir, exist_ok=True)

    requested = {t.strip() for t in args.file_types.split(",") if t.strip()}
    illegal = requested - EXTRA_FOLDERS.keys()
    if illegal:
        raise ValueError(f"Unknown file_type(s): {', '.join(sorted(illegal))}")

    folders = set(COMMON_FOLDERS)
    for t in requested:
        folders.update(EXTRA_FOLDERS[t])

    if not verify_access(DATASET_REPO):
        print(
            f"⚠  You do not have access to {DATASET_REPO}. "
            "Visit the dataset page on HuggingFace and request permission first."
        )
        return

    rel_paths = list_files(DATASET_REPO, sorted(folders))
    ok = download_files(rel_paths, args.odir, args.clean_cache, args.workers)
    print("\nDownload complete." if ok else "\nDownload finished with errors.")


if __name__ == "__main__":
    main()
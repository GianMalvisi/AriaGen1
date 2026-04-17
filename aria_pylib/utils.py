"""
aria_pylib.utils: Utility functions for Aria data workflows.
"""
import os
import glob
import pandas as pd

def infer_modality(label: str) -> str:
    """Infer a simple modality label from the stream label or name."""
    text = (label or "").lower()
    if "rgb" in text:
        return "rgb"
    if "slam" in text:
        return "slam"
    if "imu" in text:
        return "imu"
    if "mag" in text:
        return "mag"
    if "baro" in text:
        return "baro"
    if "gps" in text:
        return "gps"
    if "et" in text:
        return "et"
    if "wps" in text:
        return "wps"
    if "depth" in text:
        return "depth"
    if "image" in text:
        return "image"
    return "other"

def select_probe_indices(sample_count: int):
    """Select a compact set of representative indices for one stream."""
    if sample_count <= 0:
        return []
    candidate_indices = [
        0,
        sample_count // 4,
        sample_count // 2,
        (3 * sample_count) // 4,
        sample_count - 1,
    ]
    return sorted(set(i for i in candidate_indices if 0 <= i < sample_count))

def find_trajectory_file(slam_dir: str) -> str:
    candidates = sorted(
        glob.glob(os.path.join(slam_dir, "closed_loop_trajectory.csv")) +
        glob.glob(os.path.join(slam_dir, "closed_loop_trajectory.csv.gz"))
    )
    if not candidates:
        raise FileNotFoundError(f"No closed_loop_trajectory file found in: {slam_dir}")
    return candidates[0]

def find_points_file(slam_dir: str) -> str:
    candidates = sorted(
        glob.glob(os.path.join(slam_dir, "semidense_points.csv")) +
        glob.glob(os.path.join(slam_dir, "semidense_points.csv.gz"))
    )
    if not candidates:
        raise FileNotFoundError(f"No semidense_points file found in: {slam_dir}")
    return candidates[0]

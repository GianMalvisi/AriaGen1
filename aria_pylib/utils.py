"""
aria_pylib.utils: Recording inspection, file-discovery, and modality
classification helpers.
"""

import os
import glob


def infer_modality(label):
    """Infer a canonical modality name from a stream label string.

    Recognized modalities: rgb, slam, imu, mag, baro, gps, et, wps, depth, image.
    """
    text = (label or "").lower()
    for keyword, modality in [
        ("rgb", "rgb"), ("slam", "slam"), ("imu", "imu"), ("mag", "mag"),
        ("baro", "baro"), ("gps", "gps"), ("et", "et"), ("wps", "wps"),
        ("depth", "depth"), ("image", "image"),
    ]:
        if keyword in text:
            return modality
    return "other"


def select_probe_indices(sample_count):
    """Select up to five evenly spaced indices for stream inspection.

    Returns indices at 0%, 25%, 50%, 75%, and 100% of the stream length.
    """
    if sample_count <= 0:
        return []
    candidates = [0, sample_count // 4, sample_count // 2,
                  3 * sample_count // 4, sample_count - 1]
    return sorted(set(i for i in candidates if 0 <= i < sample_count))


def find_trajectory_file(slam_dir):
    """Locate the MPS closed-loop trajectory CSV under a SLAM output directory.

    Raises:
        FileNotFoundError: If no matching file exists.
    """
    candidates = sorted(
        glob.glob(os.path.join(slam_dir, "closed_loop_trajectory.csv"))
        + glob.glob(os.path.join(slam_dir, "closed_loop_trajectory.csv.gz"))
    )
    if not candidates:
        raise FileNotFoundError(f"No closed_loop_trajectory file in: {slam_dir}")
    return candidates[0]


def find_points_file(slam_dir):
    """Locate the MPS semi-dense point cloud CSV under a SLAM output directory.

    Raises:
        FileNotFoundError: If no matching file exists.
    """
    candidates = sorted(
        glob.glob(os.path.join(slam_dir, "semidense_points.csv"))
        + glob.glob(os.path.join(slam_dir, "semidense_points.csv.gz"))
    )
    if not candidates:
        raise FileNotFoundError(f"No semidense_points file in: {slam_dir}")
    return candidates[0]

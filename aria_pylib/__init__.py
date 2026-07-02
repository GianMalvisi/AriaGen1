"""
aria_pylib: End-to-end pipeline for gaze-driven 3D object segmentation
and isolation from Project Aria egocentric recordings.

Modules:
    loaders: PyTorch datasets for single- and multi-stream VRS access.
    sync: Stream synchronization and temporal alignment.
    utils: Recording inspection and file-discovery helpers.
    gaze: Eye-gaze projection, temporal alignment, and I-DT fixation detection.
    segmentation: SAM2 multi-run mono-prompt inference and mask I/O.
    preprocessing: Frame extraction, cropping, and COLMAP integration.
    mesh: PLY loading, face-preserving export, and sub-mesh extraction.
    projection: Pinhole + OPENCV distortion projection and majority voting.
    nerfstudio: Training checkpoint discovery and pipeline loading helpers.
"""

from .loaders import AriaDataset, AriaMultistreamDataset
from .sync import (
    resolve_stream_id,
    get_stream_timestamps_ns,
    select_rgb_reference_stream,
    build_synchronization_table,
)
from .utils import (
    infer_modality,
    select_probe_indices,
    find_trajectory_file,
    find_points_file,
)

__version__ = "0.1.0"

__all__ = [
    "AriaDataset",
    "AriaMultistreamDataset",
    "resolve_stream_id",
    "get_stream_timestamps_ns",
    "select_rgb_reference_stream",
    "build_synchronization_table",
    "infer_modality",
    "select_probe_indices",
    "find_trajectory_file",
    "find_points_file",
]

"""
aria_pylib: Python package for Aria dataloading and synchronization.
Exports main dataset classes, sync helpers, and utilities.
"""
from .loaders import AriaDataset, AriaMultistreamDataset
from .sync import resolve_stream_id, get_stream_timestamps_ns, select_rgb_reference_stream, build_synchronization_table
from .utils import infer_modality, select_probe_indices, find_trajectory_file, find_points_file

__version__ = "0.1.0"
__all__ = ["AriaDataset", "AriaMultistreamDataset"]
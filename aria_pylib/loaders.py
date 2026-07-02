"""
aria_pylib.loaders: VRS frame extraction and PyTorch dataset classes
for single-stream and synchronized multi-stream access.
"""

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .sync import resolve_stream_id, get_stream_timestamps_ns


def _candidate_to_numpy_image(candidate):
    """Recursively extract a numpy image array from a projectaria_tools payload."""
    if candidate is None:
        return None
    if isinstance(candidate, (tuple, list)):
        for item in candidate:
            arr = _candidate_to_numpy_image(item)
            if arr is not None:
                return arr
        return None
    if callable(candidate):
        try:
            return _candidate_to_numpy_image(candidate())
        except Exception:
            return None
    for attr_name in ["to_numpy_array", "to_numpy", "numpy"]:
        if hasattr(candidate, attr_name):
            try:
                value = getattr(candidate, attr_name)
                arr = value() if callable(value) else value
                arr = np.asarray(arr)
                if arr.ndim in (2, 3) and arr.size > 0:
                    return arr
            except Exception:
                pass
    if hasattr(candidate, "image_data"):
        try:
            nested = candidate.image_data() if callable(candidate.image_data) else candidate.image_data
            arr = _candidate_to_numpy_image(nested)
            if arr is not None:
                return arr
        except Exception:
            pass
    try:
        arr = np.asarray(candidate)
        if arr.ndim in (2, 3) and arr.size > 0:
            return arr
    except Exception:
        pass
    return None


def _extract_image_array_from_sample(sample):
    """Extract an RGB image array from a projectaria_tools SensorData sample."""
    for attr_name in ["image_data_and_record", "image_data", "image"]:
        if hasattr(sample, attr_name):
            value = getattr(sample, attr_name)
            candidate = value() if callable(value) else value
            arr = _candidate_to_numpy_image(candidate)
            if arr is not None:
                return arr
    raise RuntimeError(
        f"Unable to extract a valid image array from sample of type: {type(sample)}"
    )


def _numpy_to_pil_image(image_array):
    """Convert a numpy array to a PIL Image, handling channel order and dtype."""
    image_array = np.asarray(image_array)
    if image_array.ndim == 3 and image_array.shape[0] in (1, 3, 4) and image_array.shape[-1] not in (1, 3, 4):
        image_array = np.transpose(image_array, (1, 2, 0))
    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    if image_array.ndim == 2:
        return Image.fromarray(image_array, mode="L")
    if image_array.ndim == 3:
        if image_array.shape[-1] == 1:
            return Image.fromarray(image_array[..., 0], mode="L")
        if image_array.shape[-1] == 3:
            return Image.fromarray(image_array, mode="RGB")
        if image_array.shape[-1] == 4:
            return Image.fromarray(image_array, mode="RGBA")
    raise RuntimeError(f"Unsupported image shape for PIL conversion: {image_array.shape}")


def _pil_to_tensor(pil_image):
    """Convert a PIL Image to a float32 CHW tensor in [0, 1]."""
    image_array = np.array(pil_image, copy=True)
    if image_array.ndim == 2:
        image_array = image_array[..., None]
    return torch.from_numpy(image_array).permute(2, 0, 1).contiguous().float() / 255.0


def extract_rgb_frame(provider, stream_id, index):
    """Read a single RGB frame from a VRS provider by index.

    Handles API differences across projectaria_tools versions by probing
    multiple accessor paths. Returns an HxWx3 uint8 numpy array.
    """
    sample = provider.get_sensor_data_by_index(stream_id, index)
    for attr in ["image_data_and_record", "image_data", "image"]:
        if not hasattr(sample, attr):
            continue
        payload = getattr(sample, attr)
        payload = payload() if callable(payload) else payload
        if isinstance(payload, tuple):
            payload = payload[0]
        for arr_attr in ["to_numpy_array", "buffer"]:
            if hasattr(payload, arr_attr):
                arr = getattr(payload, arr_attr)
                arr = arr() if callable(arr) else arr
                arr = np.asarray(arr)
                if arr.ndim >= 2:
                    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                        arr = np.transpose(arr, (1, 2, 0))
                    return arr.astype(np.uint8)
    raise RuntimeError(f"Cannot extract image at index {index}")


def extract_frames(provider, stream_id, output_dir, subsample_stride=1, quality=95):
    """Extract RGB frames from a VRS stream and save as JPEG files.

    Args:
        provider: VRS data provider.
        stream_id: RGB stream identifier.
        output_dir: Directory to write numbered JPEG files.
        subsample_stride: Keep every Nth frame.
        quality: JPEG compression quality.

    Returns:
        indices: List of VRS-level indices that were extracted.
        n_saved: Number of frames written to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_total = provider.get_num_data(stream_id)
    indices = list(range(0, n_total, subsample_stride))
    n_frames = len(indices)

    existing = len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
    if existing >= n_frames:
        return indices, existing

    for out_idx, vrs_idx in enumerate(indices):
        frame = extract_rgb_frame(provider, stream_id, vrs_idx)
        pil_img = _numpy_to_pil_image(frame)
        pil_img.save(os.path.join(output_dir, f"{out_idx:06d}.jpg"), quality=quality)

    return indices, n_frames


class AriaDataset(Dataset):
    """PyTorch Dataset for index-based access to a single VRS stream."""

    def __init__(self, provider, stream_id, timestamps_ns, transform=None):
        self.provider = provider
        self.stream_id = stream_id
        self.timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
        self.transform = transform
        if self.provider is None:
            raise ValueError("provider must not be None")
        if len(self.timestamps_ns) == 0:
            raise ValueError("timestamps_ns must contain at least one timestamp")

    def __len__(self):
        return len(self.timestamps_ns)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = int(index.item())
        index = int(index)
        if index < 0 or index >= len(self.timestamps_ns):
            raise IndexError(f"Index {index} out of bounds for size {len(self.timestamps_ns)}")
        sample = self.provider.get_sensor_data_by_index(self.stream_id, index)
        image_array = _extract_image_array_from_sample(sample)
        pil_image = _numpy_to_pil_image(image_array)
        image = self.transform(pil_image) if self.transform else _pil_to_tensor(pil_image)
        return image, int(self.timestamps_ns[index])


class AriaMultistreamDataset(Dataset):
    """PyTorch Dataset for synchronized access to multiple VRS streams."""

    def __init__(self, provider, stream_ids, timestamps_ns, transform=None):
        self.provider = provider
        self.stream_ids = stream_ids
        self.timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
        self.transform = transform
        if self.provider is None:
            raise ValueError("provider must not be None")
        if len(self.timestamps_ns) == 0:
            raise ValueError("timestamps_ns must contain at least one timestamp")
        if not isinstance(self.stream_ids, (list, tuple)):
            raise ValueError("stream_ids must be a list or tuple")

    def __len__(self):
        return len(self.timestamps_ns)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = int(index.item())
        index = int(index)
        if index < 0 or index >= len(self.timestamps_ns):
            raise IndexError(f"Index {index} out of bounds for size {len(self.timestamps_ns)}")
        images = []
        for stream_id in self.stream_ids:
            sample = self.provider.get_sensor_data_by_index(stream_id, index)
            image_array = _extract_image_array_from_sample(sample)
            pil_image = _numpy_to_pil_image(image_array)
            image = self.transform(pil_image) if self.transform else _pil_to_tensor(pil_image)
            images.append(image)
        return images, int(self.timestamps_ns[index])

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _candidate_to_numpy_image(candidate):
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
    for attr_name in ["image_data_and_record", "image_data", "image"]:
        if hasattr(sample, attr_name):
            value = getattr(sample, attr_name)
            candidate = value() if callable(value) else value
            arr = _candidate_to_numpy_image(candidate)
            if arr is not None:
                return arr

    raise RuntimeError(f"Unable to extract a valid image array from sample of type: {type(sample)}")


def _numpy_to_pil_image(image_array):
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
    image_array = np.array(pil_image, copy=True)
    if image_array.ndim == 2:
        image_array = image_array[..., None]
    return torch.from_numpy(image_array).permute(2, 0, 1).contiguous().float() / 255.0


class AriaDataset(Dataset):
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
            raise IndexError(f"Index {index} is out of bounds for dataset of size {len(self.timestamps_ns)}")

        timestamp_ns = int(self.timestamps_ns[index])
        sample = self.provider.get_sensor_data_by_index(self.stream_id, index)

        image_array = _extract_image_array_from_sample(sample)
        pil_image = _numpy_to_pil_image(image_array)

        if self.transform is not None:
            image = self.transform(pil_image)
        else:
            image = _pil_to_tensor(pil_image)

        return image, timestamp_ns
"""
aria_pylib.segmentation: SAM2 multi-run mono-prompt video segmentation
and palettized mask I/O.
"""

import os
import numpy as np
import torch
from PIL import Image


DEFAULT_PALETTE = [
    0, 0, 0,           # 0: Background
    255, 0, 0,         # 1: Red
    0, 255, 0,         # 2: Green
    0, 100, 255,       # 3: Blue
    255, 255, 0,       # 4: Yellow
    255, 0, 255,       # 5: Magenta
    0, 255, 255,       # 6: Cyan
    255, 128, 0,       # 7: Orange
    128, 0, 255,       # 8: Purple
    0, 255, 128,       # 9: Light green
]


def run_single_prompt(predictor, inference_state, prompt):
    """Run SAM2 bidirectional propagation for a single point prompt.

    Resets the tracker state (preserving the loaded video), inserts
    one point prompt, then propagates forward to the last frame and
    backward to frame 0.

    Args:
        predictor: SAM2 video predictor instance.
        inference_state: Initialized video inference state.
        prompt: Dict with ``object_id``, ``init_frame_idx``,
                ``point_coords`` ([u, v]), ``point_labels`` ([1]).

    Returns:
        Dict mapping frame indices to boolean masks (H, W).
    """
    predictor.reset_state(inference_state)
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=prompt["init_frame_idx"],
        obj_id=prompt["object_id"],
        points=np.array([prompt["point_coords"]], dtype=np.float32),
        labels=np.array(prompt["point_labels"], dtype=np.int32),
    )

    masks = {}
    init = prompt["init_frame_idx"]

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for fi, _, ml in predictor.propagate_in_video(inference_state):
            if int(fi) >= init:
                masks[int(fi)] = (ml[0, 0] > 0).detach().cpu().numpy()
        for fi, _, ml in predictor.propagate_in_video(
            inference_state, start_frame_idx=init, reverse=True
        ):
            if int(fi) < init:
                masks[int(fi)] = (ml[0, 0] > 0).detach().cpu().numpy()

    return masks


def run_multi_prompt(predictor, inference_state, prompts):
    """Run SAM2 independently for each prompt.

    Each fixation is tracked in its own session to prevent inter-object
    interference during propagation.

    Args:
        predictor: SAM2 video predictor.
        inference_state: Video inference state (reused across runs).
        prompts: List of prompt dicts (see :func:`run_single_prompt`).

    Returns:
        Dict mapping object_id to {frame_idx: bool_mask}.
    """
    per_object_masks = {}
    for prompt in prompts:
        masks = run_single_prompt(predictor, inference_state, prompt)
        per_object_masks[prompt["object_id"]] = masks
    return per_object_masks


def merge_masks(per_object_masks):
    """Merge per-object binary masks into unified multi-label masks.

    When objects overlap, the higher object ID takes priority.

    Args:
        per_object_masks: Dict {object_id: {frame_idx: bool_mask}}.

    Returns:
        Dict {frame_idx: uint8_label_mask} with pixel values 0..N.
    """
    all_frames = set()
    for m in per_object_masks.values():
        all_frames.update(m.keys())

    sample = next(iter(next(iter(per_object_masks.values())).values()))
    h, w = sample.shape

    merged = {}
    for fi in sorted(all_frames):
        label_mask = np.zeros((h, w), dtype=np.uint8)
        for oid in sorted(per_object_masks):
            if fi in per_object_masks[oid]:
                label_mask[per_object_masks[oid][fi]] = oid
        merged[fi] = label_mask
    return merged


def save_mask(frame_idx, label_mask, output_dir, palette=None):
    """Save a label mask as a palettized PNG.

    Args:
        frame_idx: Frame index (used for filename: ``000042.png``).
        label_mask: (H, W) uint8 array with pixel values 0..N.
        output_dir: Directory to write the PNG.
        palette: Optional flat list of 768 RGB values. Defaults to
                 :data:`DEFAULT_PALETTE`.
    """
    os.makedirs(output_dir, exist_ok=True)
    if palette is None:
        palette = list(DEFAULT_PALETTE) + [0] * (768 - len(DEFAULT_PALETTE))

    img = Image.fromarray(label_mask, mode="P")
    img.putpalette(palette)
    img.save(
        os.path.join(output_dir, f"{int(frame_idx):06d}.png"),
        compress_level=1,
    )


def export_masks(merged_masks, output_dir, palette=None):
    """Export all merged label masks to disk as palettized PNGs.

    Args:
        merged_masks: Dict {frame_idx: uint8_label_mask}.
        output_dir: Target directory.
        palette: Optional palette (see :func:`save_mask`).

    Returns:
        Number of masks written.
    """
    for fi, mask in sorted(merged_masks.items()):
        save_mask(fi, mask, output_dir, palette)
    return len(merged_masks)


def load_mask(mask_path):
    """Load a palettized PNG mask preserving label indices.

    Uses PIL to read palette indices directly (0, 1, 2, ...) rather than
    decoded RGB values, which is critical for palettized masks.

    Args:
        mask_path: Path to the PNG file.

    Returns:
        (H, W) uint8 numpy array of label indices.
    """
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask

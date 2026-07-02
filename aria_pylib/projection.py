"""
aria_pylib.projection: Pinhole + OPENCV radial-tangential projection,
mask-to-mesh projection, and majority voting for semantic labelling.
"""

import os
import numpy as np
import cv2
from PIL import Image


def project_with_distortion(xyz, c2w_3x4, fx, fy, cx, cy, k1, k2, p1, p2, W, H):
    """Project 3D points onto a camera with OpenGL convention and OPENCV
    radial-tangential distortion.

    Convention: camera looks along -Z, +Y is up.

    Args:
        xyz: (N, 3) world-space vertex positions.
        c2w_3x4: (3, 4) camera-to-world matrix (from nerfstudio).
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point in pixels.
        k1, k2: Radial distortion coefficients.
        p1, p2: Tangential distortion coefficients.
        W, H: Image width and height.

    Returns:
        uv: (N, 2) int32 pixel coordinates.
        valid: (N,) bool mask, True if the point is in front of the camera
                and projects within image bounds.
    """
    c2w = np.eye(4)
    c2w[:3, :] = c2w_3x4
    w2c = np.linalg.inv(c2w)
    cam = (w2c @ np.hstack([xyz, np.ones((len(xyz), 1))]).T).T
    Xc, Yc, Zc = cam[:, 0], cam[:, 1], cam[:, 2]

    depth = -Zc
    ok = depth > 1e-4
    sd = np.where(ok, depth, 1.0)
    xn = Xc / sd
    yn = -Yc / sd

    r2 = xn ** 2 + yn ** 2
    r4 = r2 ** 2
    rad = 1 + k1 * r2 + k2 * r4
    xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn ** 2)
    yd = yn * rad + p1 * (r2 + 2 * yn ** 2) + 2 * p2 * xn * yn

    u = (fx * xd + cx).astype(np.int32)
    v = (fy * yd + cy).astype(np.int32)
    valid = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return np.stack([u, v], axis=1), valid


def find_ns_camera(pipeline, frame_name):
    """Search the nerfstudio pipeline for a camera matching ``frame_name``.

    Scans both the train and eval datasets.

    Args:
        pipeline: Loaded nerfstudio pipeline (from ``eval_setup``).
        frame_name: Substring to match against image filenames..

    Returns:
        (camera, split, index) or (None, None, None) if not found.
    """
    for split in ["train", "eval"]:
        ds = getattr(pipeline.datamanager, f"{split}_dataset", None)
        if ds is None:
            continue
        for i, fn in enumerate(ds._dataparser_outputs.image_filenames):
            if frame_name in str(fn):
                return ds.cameras[i: i + 1], split, i
    return None, None, None


def load_mask_for_projection(mask_path, W, H):
    """Load a palettized PNG mask and center-crop to match the nerfstudio
    image resolution.

    Aria masks are stored at the original sensor resolution
    while nerfstudio images are cropped. The crop offset 
    is computed as a centered extraction.

    Args:
        mask_path: Path to the palettized PNG mask.
        W: Target width (nerfstudio image width).
        H: Target height (nerfstudio image height).

    Returns:
        (H, W) uint8 array of label indices.
    """
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mh, mw = mask.shape
    ox = (mw - W) // 2
    oy = (mh - H) // 2
    if ox >= 0 and oy >= 0:
        return mask[oy: oy + H, ox: ox + W]
    return cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)


def _extract_camera_params(cam):
    """Extract projection parameters from a nerfstudio Cameras object."""
    c2w = cam.camera_to_worlds[0].cpu().numpy()
    fx = cam.fx[0].item()
    fy = cam.fy[0].item()
    cx = cam.cx[0].item()
    cy = cam.cy[0].item()
    H = int(cam.height[0].item())
    W = int(cam.width[0].item())
    dp = (
        cam.distortion_params[0].cpu().numpy()
        if cam.distortion_params is not None
        else np.zeros(6)
    )
    k1, k2, p1, p2 = float(dp[0]), float(dp[1]), float(dp[2]), float(dp[3])
    return c2w, fx, fy, cx, cy, k1, k2, p1, p2, W, H


def project_masks_onto_mesh(xyz_mesh, pipeline, masks_dir, mask_files=None):
    """Project all available masks onto mesh vertices and accumulate per-vertex
    vote counts.

    For each mask, the corresponding nerfstudio camera is located via
    ``find_ns_camera``, the mask is loaded and center-cropped, and every
    visible vertex is looked up in the mask to record its label.

    Args:
        xyz_mesh:( N, 3) mesh vertex positions.
        pipeline: Loaded nerfstudio pipeline.
        masks_dir: Directory containing palettized PNG masks.
        mask_files: Optional explicit list of mask filenames. If None,
                    all ``*.png`` files in ``masks_dir`` are used.

    Returns:
        n_views_seen: (N,) int32 — frames where each vertex was visible.
        n_subject: (N,) int32 — frames where each vertex was labelled > 0.
        votes_per_label: Dict {label_id: (N,) int32 vote array}.
    """
    if mask_files is None:
        mask_files = sorted(f for f in os.listdir(masks_dir) if f.endswith(".png"))

    N = len(xyz_mesh)
    n_views_seen = np.zeros(N, dtype=np.int32)
    n_subject = np.zeros(N, dtype=np.int32)
    votes_per_label = {}

    for mask_name in mask_files:
        mask_id = int(mask_name.replace(".png", ""))
        frame_name = f"frame_{mask_id + 1:05d}"
        mask_path = os.path.join(masks_dir, mask_name)

        cam, _, _ = find_ns_camera(pipeline, frame_name)
        if cam is None:
            continue

        c2w, fx, fy, cx, cy, k1, k2, p1, p2, W, H = _extract_camera_params(cam)
        mask = load_mask_for_projection(mask_path, W, H)
        uv, valid = project_with_distortion(
            xyz_mesh, c2w, fx, fy, cx, cy, k1, k2, p1, p2, W, H
        )
        valid_idx = np.where(valid)[0]

        n_views_seen[valid_idx] += 1
        labels = mask[uv[valid_idx, 1], uv[valid_idx, 0]]
        n_subject[valid_idx[labels > 0]] += 1

        for lbl in np.unique(labels):
            if lbl == 0:
                continue
            lbl = int(lbl)
            if lbl not in votes_per_label:
                votes_per_label[lbl] = np.zeros(N, dtype=np.int32)
            votes_per_label[lbl][valid_idx[labels == lbl]] += 1

    return n_views_seen, n_subject, votes_per_label


def majority_vote(n_views_seen, n_subject, votes_per_label, threshold_pct=66.6):
    """Apply percentage-based supermajority voting to classify mesh vertices.

    A vertex is classified as subject if the ratio
    ``n_subject / n_views_seen >= threshold_pct / 100``.

    Args:
        n_views_seen: (N,) int32 — eligible frame count per vertex.
        n_subject: (N,) int32 — subject-labelled frame count per vertex.
        votes_per_label: Dict {label_id: (N,) int32}.
        threshold_pct: Minimum subject ratio as a percentage (default 66.6%).

    Returns:
        passes: (N,) bool, True for vertices passing the vote.
        best_label: (N,) int32, winning label for each passing vertex (0 otherwise).
    """
    N = len(n_views_seen)
    safe = np.maximum(n_views_seen, 1)
    ratio = n_subject / safe
    passes = (ratio >= threshold_pct / 100.0) & (n_subject > 0)

    best_label = np.zeros(N, dtype=np.int32)
    for lbl in sorted(votes_per_label.keys()):
        is_best = passes.copy()
        for other in votes_per_label:
            if other != lbl:
                is_best &= votes_per_label[lbl] >= votes_per_label.get(
                    other, np.zeros(N, dtype=np.int32)
                )
        best_label[is_best] = lbl

    return passes, best_label

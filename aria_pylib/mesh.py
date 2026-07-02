"""
aria_pylib.mesh: PLY mesh I/O and sub-mesh extraction with face remapping.
"""

import os
import numpy as np
from plyfile import PlyData, PlyElement


def load_ply(path):
    """Load a PLY file and extract vertex positions, colors, and the raw PlyData.

    Args:
        path: Path to the ``.ply`` file.

    Returns:
        xyz: (N, 3) float64 vertex positions.
        rgb: (N, 3) uint8 vertex colors, or None if absent.
        ply: The original PlyData object (for face preservation).
    """
    ply = PlyData.read(path)
    v = ply["vertex"]
    xyz = np.stack(
        [np.array(v["x"], np.float64),
         np.array(v["y"], np.float64),
         np.array(v["z"], np.float64)],
        axis=1,
    )
    props = {p.name for p in v.properties}
    rgb = None
    if {"red", "green", "blue"}.issubset(props):
        rgb = np.stack(
            [np.array(v["red"], np.uint8),
             np.array(v["green"], np.uint8),
             np.array(v["blue"], np.uint8)],
            axis=1,
        ).copy()
    return xyz, rgb, ply


def save_ply(path, ply_orig, rgb_new):
    """Save a PLY with updated vertex colors while preserving face topology.

    Args:
        path: Output file path.
        ply_orig: Original PlyData (provides vertex dtype and face element).
        rgb_new: (N, 3) uint8 array of new vertex colors.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    v = ply_orig["vertex"]
    nd = np.empty(len(v.data), dtype=v.data.dtype)
    nd[:] = v.data[:]
    if "red" in {p.name for p in v.properties}:
        nd["red"] = rgb_new[:, 0]
        nd["green"] = rgb_new[:, 1]
        nd["blue"] = rgb_new[:, 2]
    elems = [PlyElement.describe(nd, "vertex")]
    for e in ply_orig.elements:
        if e.name != "vertex":
            elems.append(e)
    PlyData(elems).write(path)


def extract_submesh(ply_orig, vertex_mask):
    """Extract a sub-mesh containing only selected vertices.

    Retains only faces whose three vertices all satisfy ``vertex_mask``.
    Face vertex indices are remapped to the new compact vertex array.

    Args:
        ply_orig: Original PlyData object.
        vertex_mask: (N,) boolean array, True for vertices to keep.

    Returns:
        PlyData object containing the filtered vertices and remapped faces.
    """
    v = ply_orig["vertex"]
    N = len(v.data)

    idx_map = np.full(N, -1, dtype=np.int32)
    idx_map[vertex_mask] = np.arange(vertex_mask.sum(), dtype=np.int32)
    new_v = v.data[vertex_mask]
    elems = [PlyElement.describe(new_v, "vertex")]

    if "face" in [e.name for e in ply_orig.elements]:
        fd = ply_orig["face"].data
        kept = []
        for fv in fd["vertex_indices"]:
            rm = idx_map[fv]
            if np.all(rm >= 0):
                kept.append(rm)
        nf = np.empty(len(kept), dtype=[("vertex_indices", "O")])
        for i, fv in enumerate(kept):
            nf["vertex_indices"][i] = np.array(fv, dtype=np.int32)
        elems.append(PlyElement.describe(nf, "face"))

    return PlyData(elems)

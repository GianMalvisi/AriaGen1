"""
aria_pylib.preprocessing: Image cropping for Aria lens distortion removal
and COLMAP pose estimation via nerfstudio.
"""

import os
import subprocess
import sys
from PIL import Image


def apply_crop(input_dir, output_dir, extension=".jpg", crop_percent=0.15):
    """Crop a fixed percentage from all edges of each image.

    Aria Gen 1 images exhibit radial distortion artifacts near the border.
    Removing the outer ``crop_percent`` on each side produces cleaner inputs
    for Structure-from-Motion.

    Args:
        input_dir: Directory containing the source images.
        output_dir: Directory to write the cropped images.
        extension: File extension filter (case-insensitive).
        crop_percent: Fraction of each dimension to remove per side (default 15%).

    Returns:
        Number of images cropped.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(extension))

    for fn in files:
        with Image.open(os.path.join(input_dir, fn)) as img:
            w, h = img.size
            box = (
                int(w * crop_percent),
                int(h * crop_percent),
                int(w * (1 - crop_percent)),
                int(h * (1 - crop_percent)),
            )
            img.crop(box).save(os.path.join(output_dir, fn))

    return len(files)


def run_colmap(images_dir, output_dir):
    """Run COLMAP via ``ns-process-data images`` to estimate camera poses.

    Args:
        images_dir: Path to the cropped image directory.
        output_dir: Nerfstudio project root (will contain ``transforms.json``).

    Raises:
        RuntimeError: If the subprocess exits with a non-zero return code.
    """
    cmd = ["ns-process-data", "images", "--data", images_dir, "--output-dir", output_dir]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )
    for line in iter(proc.stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ns-process-data failed with exit code {proc.returncode}")

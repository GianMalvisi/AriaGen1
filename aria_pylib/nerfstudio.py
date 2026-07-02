"""
aria_pylib.nerfstudio: Helpers for discovering training checkpoints,
loading the nerfstudio pipeline with embedding-size mismatch handling,
and running training / export commands.
"""

import os
import sys
import subprocess
import torch
from pathlib import Path


def find_latest_config(training_root):
    """Locate the most recent ``config.yml`` under a nerfstudio training directory.

    Args:
        training_root: Root directory to search recursively.

    Returns:
        Absolute path to the newest ``config.yml``.

    Raises:
        FileNotFoundError: If no ``config.yml`` is found.
    """
    configs = sorted(
        Path(training_root).rglob("config.yml"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not configs:
        raise FileNotFoundError(f"No config.yml found under {training_root}")
    return str(configs[0].resolve())


def load_pipeline(config_path, patch_embeddings=True):
    """Load a nerfstudio pipeline from a training config.

    If the checkpoint was trained on a different number of images than
    currently present in the data directory, the appearance embeddings 
    and camera optimizer tensors will have mismatched shapes.
    When ``patch_embeddings=True``, mismatched tensors are 
    resized automatically to allow loading.

    Args:
        config_path:       Path to ``config.yml``.
        patch_embeddings:  If True, resize mismatched state-dict tensors
                           before loading (default True).

    Returns:
        (config, pipeline, checkpoint_path, step)
    """
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.pipelines.base_pipeline import Pipeline

    if patch_embeddings:
        _original_load = Pipeline.load_state_dict

        def _patched_load(self, state_dict, strict=True):
            current = self.state_dict()
            for key in list(state_dict.keys()):
                if key in current and state_dict[key].shape != current[key].shape:
                    old_shape = state_dict[key].shape
                    new_shape = current[key].shape
                    new_tensor = torch.zeros(new_shape, dtype=state_dict[key].dtype)
                    slices = tuple(
                        slice(0, min(o, n)) for o, n in zip(old_shape, new_shape)
                    )
                    new_tensor[slices] = state_dict[key][slices]
                    state_dict[key] = new_tensor
            return _original_load(self, state_dict, strict=False)

        Pipeline.load_state_dict = _patched_load

    result = eval_setup(Path(config_path))

    if patch_embeddings:
        Pipeline.load_state_dict = _original_load

    return result


def run_training(data_dir, output_dir, experiment_name="nerfacto", timestamp=None,
                 extra_args=None):
    """Launch ``ns-train nerfacto`` as a subprocess.

    Args:
        data_dir: Nerfstudio project root (containing ``transforms.json``).
        output_dir: Directory for training outputs.
        experiment_name: Experiment name for the output subdirectory.
        timestamp: Optional timestamp string for the run directory.
        extra_args: Optional list of additional CLI arguments.

    Raises:
        RuntimeError: If training exits with a non-zero return code.
    """
    cmd = [
        "ns-train", "nerfacto",
        "--data", data_dir,
        "--output-dir", output_dir,
        "--experiment-name", experiment_name,
    ]
    if timestamp:
        cmd += ["--timestamp", timestamp]
    if extra_args:
        cmd += extra_args
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
        raise RuntimeError(f"ns-train failed with exit code {proc.returncode}")


def export_mesh(config_path, output_dir, target_num_faces=50000,
                num_points=1000000, extra_args=None):
    """Export a Poisson mesh from a trained nerfstudio checkpoint.

    Args:
        config_path: Path to the training ``config.yml``.
        output_dir: Directory for the exported PLY.
        target_num_faces: Target face count for the Poisson reconstruction.
        num_points: Number of points to sample for the reconstruction.
        extra_args: Optional list of additional CLI arguments.

    Returns:
        Path to the first ``.ply`` file found in the output directory.

    Raises:
        RuntimeError: If export fails.
        FileNotFoundError: If no PLY is produced.
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ns-export", "poisson",
        "--load-config", config_path,
        "--output-dir", output_dir,
        "--target-num-faces", str(target_num_faces),
        "--num-points", str(num_points),
        "--num-pixels-per-side", "2048",
        "--remove-outliers", "True",
        "--normal-method", "open3d",
    ]
    if extra_args:
        cmd += extra_args
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
        raise RuntimeError(f"ns-export failed with exit code {proc.returncode}")

    candidates = list(Path(output_dir).rglob("*.ply"))
    if not candidates:
        raise FileNotFoundError(f"No PLY file found in {output_dir}")
    return str(candidates[0])

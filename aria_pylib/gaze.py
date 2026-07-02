"""
aria_pylib.gaze: Eye-gaze projection, temporal alignment with RGB frames,
and I-DT dispersion-based fixation detection.
"""

import numpy as np
import pandas as pd


def project_gaze_to_rgb(yaw_rads, pitch_rads, rgb_calib, T_device_cpf, T_rgb_device):
    """Project a gaze direction from CPF (Central Pupil Frame) to a 2D pixel
    on the RGB camera image plane.

    Args:
        yaw_rads:       Horizontal gaze angle in radians (CPF).
        pitch_rads:     Vertical gaze angle in radians (CPF).
        rgb_calib:      RGB camera calibration object (projectaria_tools).
        T_device_cpf:   Transform from CPF to device frame.
        T_rgb_device:   Transform from device frame to RGB camera frame.

    Returns:
        Projected pixel coordinates as a numpy array [u, v], or None on failure.
    """
    gaze_vector_cpf = np.array([np.tan(yaw_rads), np.tan(pitch_rads), 1.0])
    gaze_vector_device = T_device_cpf @ gaze_vector_cpf
    gaze_vector_rgb = T_rgb_device @ gaze_vector_device
    return rgb_calib.project(gaze_vector_rgb)


def align_gaze_to_rgb(rgb_timestamps_ns, gaze_timestamps_ns, eye_gazes,
                       get_yaw, get_pitch, max_delta_ms=60.0):
    """Align the MPS eye-gaze stream to RGB frame timestamps via
    nearest-neighbour matching.

    The gaze stream and the RGB stream operate at different rates. 
    For each RGB timestamp, the closest gaze sample is selected.
     Matches exceeding ``max_delta_ms`` are flagged as
    unreliable.

    Args:
        rgb_timestamps_ns: (N,) int64 array of RGB frame timestamps in nanoseconds.
        gaze_timestamps_ns: (M,) int64 array of gaze sample timestamps in nanoseconds.
        eye_gazes: List of MPS eye-gaze objects.
        get_yaw: Callable extracting yaw (radians) from a gaze object.
        get_pitch: Callable extracting pitch (radians) from a gaze object.
        max_delta_ms: Maximum acceptable time delta for a reliable match.

    Returns:
        pd.DataFrame with columns: frame_idx, rgb_ts_ns, gaze_idx, delta_ms,
        reliable, yaw_rads, pitch_rads.
    """
    n_gaze = len(gaze_timestamps_ns)
    insert_pos = np.clip(np.searchsorted(gaze_timestamps_ns, rgb_timestamps_ns), 1, n_gaze - 1)
    delta_before = np.abs(rgb_timestamps_ns - gaze_timestamps_ns[insert_pos - 1])
    delta_after = np.abs(rgb_timestamps_ns - gaze_timestamps_ns[insert_pos])
    nearest_idx = np.where(delta_before < delta_after, insert_pos - 1, insert_pos)
    nearest_delta_ms = np.minimum(delta_before, delta_after) / 1e6
    reliable = nearest_delta_ms < max_delta_ms

    return pd.DataFrame({
        "frame_idx": np.arange(len(rgb_timestamps_ns)),
        "rgb_ts_ns": rgb_timestamps_ns,
        "gaze_idx": nearest_idx,
        "delta_ms": nearest_delta_ms,
        "reliable": reliable,
        "yaw_rads": [get_yaw(eye_gazes[i]) for i in nearest_idx],
        "pitch_rads": [get_pitch(eye_gazes[i]) for i in nearest_idx],
    })


def project_all_gaze(alignment_df, rgb_calib, T_device_cpf, T_rgb_device, img_w, img_h):
    """Project gaze directions onto the RGB image plane for all aligned frames.

    Appends ``gaze_x``, ``gaze_y``, and ``valid_gaze`` columns to the
    alignment DataFrame (in-place) and returns it.

    Args:
        alignment_df: DataFrame produced by :func:`align_gaze_to_rgb`.
        rgb_calib: RGB camera calibration object.
        T_device_cpf: CPF-to-device transform.
        T_rgb_device: Device-to-RGB-camera transform.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        The input DataFrame with three new columns added.
    """
    gx, gy, valid = [], [], []
    for _, row in alignment_df.iterrows():
        if pd.isna(row["yaw_rads"]) or pd.isna(row["pitch_rads"]) or not row["reliable"]:
            gx.append(np.nan); gy.append(np.nan); valid.append(False)
            continue
        px = project_gaze_to_rgb(row["yaw_rads"], row["pitch_rads"],
                                  rgb_calib, T_device_cpf, T_rgb_device)
        if px is not None:
            u, v = float(px[0]), float(px[1])
            ok = (0 <= u < img_w) and (0 <= v < img_h)
            gx.append(u); gy.append(v); valid.append(ok)
        else:
            gx.append(np.nan); gy.append(np.nan); valid.append(False)

    alignment_df["gaze_x"] = gx
    alignment_df["gaze_y"] = gy
    alignment_df["valid_gaze"] = valid
    return alignment_df


def detect_fixations_idt(alignment_df, dispersion_threshold, min_duration_frames):
    """Detect gaze fixations using the I-DT (Identification by Dispersion
    Threshold) algorithm.

    Uses Manhattan dispersion ``(dx + dy)`` and median centroid for
    robustness against outliers at fixation boundaries.

    Args:
        alignment_df: DataFrame with ``valid_gaze``, ``gaze_x``,
                     ``gaze_y``, and ``frame_idx`` columns.
        dispersion_threshold: Maximum allowed Manhattan dispersion in pixels.
        min_duration_frames: Minimum fixation duration in frames.

    Returns:
        List of dicts, each containing ``start_frame``, ``end_frame``,
        ``centroid_x``, ``centroid_y``, and ``duration_frames``.
    """
    fixations = []
    valid_mask = alignment_df["valid_gaze"].values
    xs = alignment_df["gaze_x"].values
    ys = alignment_df["gaze_y"].values
    frames = alignment_df["frame_idx"].values
    n = len(frames)
    i = 0

    while i <= n - min_duration_frames:
        if not all(valid_mask[i: i + min_duration_frames]):
            i += 1
            continue

        wx = xs[i: i + min_duration_frames].tolist()
        wy = ys[i: i + min_duration_frames].tolist()
        disp = (max(wx) - min(wx)) + (max(wy) - min(wy))

        if disp <= dispersion_threshold:
            j = i + min_duration_frames
            while j < n and valid_mask[j]:
                wx.append(xs[j])
                wy.append(ys[j])
                if (max(wx) - min(wx)) + (max(wy) - min(wy)) > dispersion_threshold:
                    wx.pop()
                    wy.pop()
                    break
                j += 1

            fixations.append({
                "start_frame": int(frames[i]),
                "end_frame": int(frames[j - 1]),
                "centroid_x": float(np.median(wx)),
                "centroid_y": float(np.median(wy)),
                "duration_frames": j - i,
            })
            i = j
        else:
            i += 1

    return fixations


def fixations_to_prompts(df_fixations):
    """Convert a fixation DataFrame to a list of SAM2-compatible prompt dicts.

    Each prompt uses the mid-frame of the fixation window as the initialization
    frame and the median centroid as the point coordinate.

    Args:
        df_fixations: DataFrame with ``object_id``, ``start_frame``,
                       ``end_frame``, ``centroid_x``, ``centroid_y``.

    Returns:
        List of dicts with keys ``object_id``, ``init_frame_idx``,
        ``point_coords``, ``point_labels``.
    """
    prompts = []
    for _, row in df_fixations.iterrows():
        mid_frame = (int(row["start_frame"]) + int(row["end_frame"])) // 2
        prompts.append({
            "object_id": int(row["object_id"]),
            "init_frame_idx": mid_frame,
            "point_coords": [float(row["centroid_x"]), float(row["centroid_y"])],
            "point_labels": [1],
        })
    return prompts

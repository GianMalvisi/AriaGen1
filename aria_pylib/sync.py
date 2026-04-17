"""
aria_pylib.sync: Synchronization and stream alignment helpers for Aria data.
"""
import numpy as np
import pandas as pd

def resolve_stream_id(provider, stream_name_or_id):
    """Resolve a stream name or ID to a canonical stream ID using the provider."""
    if isinstance(stream_name_or_id, int):
        return stream_name_or_id
    if hasattr(provider, 'get_stream_id_by_name'):
        return provider.get_stream_id_by_name(stream_name_or_id)
    return stream_name_or_id

def get_stream_timestamps_ns(provider, stream_id):
    """Get all timestamps (ns) for a given stream from the provider."""
    if hasattr(provider, 'get_all_timestamps_ns'):
        return provider.get_all_timestamps_ns(stream_id)
    raise ValueError(f"Provider does not support get_all_timestamps_ns for stream {stream_id}")

def select_rgb_reference_stream(streams_df):
    """Select the first stream whose label contains 'rgb' (case-insensitive), else first image, else first available."""
    rgb_candidates = streams_df[streams_df["label"].str.lower().str.contains("rgb")]
    if not rgb_candidates.empty:
        return rgb_candidates.iloc[0]["stream_id"]
    image_candidates = streams_df[streams_df["type"] == "image"]
    if not image_candidates.empty:
        return image_candidates.iloc[0]["stream_id"]
    return streams_df.iloc[0]["stream_id"] if not streams_df.empty else None

def build_synchronization_table(provider, streams_df, reference_stream_id):
    """Align all streams to the reference stream by nearest timestamp."""
    ref_stream_obj = resolve_stream_id(provider, reference_stream_id)
    ref_timestamps = get_stream_timestamps_ns(provider, ref_stream_obj)
    sync_rows = []
    for _, stream_row in streams_df.iterrows():
        stream_id = stream_row["stream_id"]
        stream_obj = resolve_stream_id(provider, stream_id)
        if stream_obj is None:
            sync_rows.append({
                "stream_id": stream_id,
                "label": stream_row["label"],
                "type": stream_row["type"],
                "aligned_indices": [None] * len(ref_timestamps),
                "aligned_timestamps_ns": [None] * len(ref_timestamps),
                "num_aligned": 0,
                "status": "UNRESOLVABLE",
            })
            continue
        stream_timestamps = get_stream_timestamps_ns(provider, stream_obj)
        aligned_indices = []
        aligned_timestamps = []
        for t_ref in ref_timestamps:
            idx = np.searchsorted(stream_timestamps, t_ref, side="left")
            if idx == 0:
                idx_aligned = 0
            elif idx == len(stream_timestamps):
                idx_aligned = len(stream_timestamps) - 1
            else:
                before = stream_timestamps[idx - 1]
                after = stream_timestamps[idx]
                idx_aligned = idx - 1 if abs(t_ref - before) <= abs(t_ref - after) else idx
            t_aligned = stream_timestamps[idx_aligned] if len(stream_timestamps) > 0 else None
            aligned_indices.append(idx_aligned if len(stream_timestamps) > 0 else None)
            aligned_timestamps.append(t_aligned)
        sync_rows.append({
            "stream_id": stream_id,
            "label": stream_row["label"],
            "type": stream_row["type"],
            "aligned_indices": aligned_indices,
            "aligned_timestamps_ns": aligned_timestamps,
            "num_aligned": len(aligned_indices),
            "status": "OK" if stream_obj is not None else "UNRESOLVABLE",
        })
    sync_df = pd.DataFrame(sync_rows)
    return sync_df

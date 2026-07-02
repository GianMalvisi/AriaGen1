"""
aria_pylib.sync: Stream resolution and temporal synchronization for
multi-stream Aria VRS recordings.
"""

import numpy as np
import pandas as pd


def resolve_stream_id(provider, stream_name_or_id):
    """Resolve a stream name or numeric ID to a canonical stream ID."""
    if isinstance(stream_name_or_id, int):
        return stream_name_or_id
    if hasattr(provider, "get_stream_id_by_name"):
        return provider.get_stream_id_by_name(stream_name_or_id)
    return stream_name_or_id


def get_stream_timestamps_ns(provider, stream_id):
    """Retrieve all timestamps in nanoseconds for a given stream.

    Raises:
        ValueError: If the provider does not expose timestamp access.
    """
    if hasattr(provider, "get_all_timestamps_ns"):
        return provider.get_all_timestamps_ns(stream_id)
    raise ValueError(f"Provider does not support get_all_timestamps_ns for stream {stream_id}")


def select_rgb_reference_stream(streams_df):
    """Select the primary RGB stream from a stream inventory DataFrame.

    Priority: streams labelled 'rgb' > any image stream > first available.
    """
    rgb = streams_df[streams_df["label"].str.lower().str.contains("rgb")]
    if not rgb.empty:
        return rgb.iloc[0]["stream_id"]
    images = streams_df[streams_df["type"] == "image"]
    if not images.empty:
        return images.iloc[0]["stream_id"]
    return streams_df.iloc[0]["stream_id"] if not streams_df.empty else None


def build_synchronization_table(provider, streams_df, reference_stream_id):
    """Align all streams to a reference stream by nearest timestamp.

    For each reference timestamp, the closest sample in every other stream
    is identified via binary search.

    Args:
        provider: VRS data provider.
        streams_df: DataFrame with ``stream_id``, ``label``, ``type``.
        reference_stream_id: Stream ID to use as the temporal reference.

    Returns:
        DataFrame with per-stream alignment metadata.
    """
    ref_obj = resolve_stream_id(provider, reference_stream_id)
    ref_ts = get_stream_timestamps_ns(provider, ref_obj)
    rows = []

    for _, sr in streams_df.iterrows():
        stream_obj = resolve_stream_id(provider, sr["stream_id"])
        if stream_obj is None:
            rows.append({
                "stream_id": sr["stream_id"], "label": sr["label"],
                "type": sr["type"], "aligned_indices": [None] * len(ref_ts),
                "aligned_timestamps_ns": [None] * len(ref_ts),
                "num_aligned": 0, "status": "UNRESOLVABLE",
            })
            continue

        stream_ts = get_stream_timestamps_ns(provider, stream_obj)
        a_idx, a_ts = [], []

        for t in ref_ts:
            idx = np.searchsorted(stream_ts, t, side="left")
            if idx == 0:
                best = 0
            elif idx >= len(stream_ts):
                best = len(stream_ts) - 1
            else:
                best = idx - 1 if abs(t - stream_ts[idx - 1]) <= abs(t - stream_ts[idx]) else idx
            a_idx.append(best if len(stream_ts) else None)
            a_ts.append(stream_ts[best] if len(stream_ts) else None)

        rows.append({
            "stream_id": sr["stream_id"], "label": sr["label"],
            "type": sr["type"], "aligned_indices": a_idx,
            "aligned_timestamps_ns": a_ts,
            "num_aligned": len(a_idx), "status": "OK",
        })

    return pd.DataFrame(rows)

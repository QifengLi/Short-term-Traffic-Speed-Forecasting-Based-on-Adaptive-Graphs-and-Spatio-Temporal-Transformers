from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_adj_from_pkl(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        content = pickle.load(f, encoding="latin1")
    if isinstance(content, (tuple, list)) and len(content) >= 3:
        adj = content[2]
    elif isinstance(content, np.ndarray):
        adj = content
    else:
        raise ValueError(f"Unsupported adj pkl structure: {type(content)}")
    adj = np.asarray(adj, dtype=np.float32)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency must be square, got {adj.shape}")
    return adj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traffic-h5",
        type=Path,
        default=Path("data/raw/metr-la.h5"),
        help="METR-LA HDF5 file (pandas DataFrame).",
    )
    parser.add_argument(
        "--adj-pkl",
        type=Path,
        default=Path("data/raw/adj_mx.pkl"),
        help="Adjacency pkl from DCRNN sensor_graph.",
    )
    parser.add_argument("--output-data", type=Path, default=Path("data/metr_la.npz"))
    parser.add_argument("--output-adj", type=Path, default=Path("data/metr_la_adj.npy"))
    parser.add_argument("--max-timesteps", type=int, default=0)
    args = parser.parse_args()

    if not args.traffic_h5.exists():
        raise FileNotFoundError(
            f"METR-LA file not found: {args.traffic_h5}. "
            "Please download metr-la.h5 from the official DCRNN data links and place it under data/raw/."
        )
    if not args.adj_pkl.exists():
        raise FileNotFoundError(f"Adjacency pkl not found: {args.adj_pkl}")

    df = pd.read_hdf(args.traffic_h5)
    values = df.values.astype(np.float32)  # [T, N]
    if args.max_timesteps > 0:
        values = values[: args.max_timesteps]

    data = values[:, :, None]  # [T, N, 1]
    adj = load_adj_from_pkl(args.adj_pkl)

    if adj.shape[0] != data.shape[1]:
        raise ValueError(f"Node mismatch: data N={data.shape[1]}, adj shape={adj.shape}")

    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    args.output_adj.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_data, data=data)
    np.save(args.output_adj, adj.astype(np.float32))

    density = float(np.count_nonzero(adj) / adj.size)
    print(f"Saved data to: {args.output_data} shape={data.shape}")
    print(
        f"Saved adj to: {args.output_adj} shape={adj.shape} nonzero={int(np.count_nonzero(adj))} density={density:.4f}"
    )


if __name__ == "__main__":
    main()

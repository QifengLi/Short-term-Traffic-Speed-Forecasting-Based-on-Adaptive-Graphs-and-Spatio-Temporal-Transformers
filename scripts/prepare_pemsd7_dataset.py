from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def normalize_distance_to_adjacency(
    dist: np.ndarray,
    sigma: float | None = None,
    threshold: float = 0.1,
) -> np.ndarray:
    # dist: [N, N], 0 on diagonal and possibly for disconnected pairs.
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError(f"Distance matrix must be square, got {dist.shape}")

    positive = dist[dist > 0]
    if positive.size == 0:
        raise ValueError("Distance matrix has no positive entries.")

    if sigma is None:
        sigma = float(np.std(positive))
    sigma = max(float(sigma), 1e-6)

    adj = np.exp(-((dist / sigma) ** 2))
    adj[dist <= 0] = 0.0
    adj[adj < threshold] = 0.0
    np.fill_diagonal(adj, 0.0)
    return adj.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v-csv",
        type=Path,
        default=Path("data/raw/PeMSD7_Full/PeMSD7_V_228.csv"),
        help="Traffic speed matrix CSV with shape [T, N].",
    )
    parser.add_argument(
        "--w-csv",
        type=Path,
        default=Path("data/raw/PeMSD7_Full/PeMSD7_W_228.csv"),
        help="Distance matrix CSV with shape [N, N].",
    )
    parser.add_argument("--output-data", type=Path, default=Path("data/pemsd7_228.npz"))
    parser.add_argument("--output-adj", type=Path, default=Path("data/pemsd7_228_adj.npy"))
    parser.add_argument("--adj-threshold", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=-1.0, help="Gaussian sigma for distance kernel.")
    parser.add_argument("--max-timesteps", type=int, default=0, help="Optional truncation for faster experiments.")
    args = parser.parse_args()

    if not args.v_csv.exists():
        raise FileNotFoundError(f"Traffic CSV not found: {args.v_csv}")
    if not args.w_csv.exists():
        raise FileNotFoundError(f"Distance CSV not found: {args.w_csv}")

    speed = np.loadtxt(args.v_csv, delimiter=",").astype(np.float32)  # [T, N]
    dist = np.loadtxt(args.w_csv, delimiter=",").astype(np.float32)  # [N, N]

    if speed.ndim != 2:
        raise ValueError(f"Expected speed shape [T, N], got {speed.shape}")
    if dist.shape[0] != speed.shape[1] or dist.shape[1] != speed.shape[1]:
        raise ValueError(
            f"Node mismatch between speed and distance: speed N={speed.shape[1]}, distance shape={dist.shape}"
        )

    if args.max_timesteps > 0:
        speed = speed[: args.max_timesteps]

    sigma = None if args.sigma <= 0 else args.sigma
    adj = normalize_distance_to_adjacency(dist, sigma=sigma, threshold=args.adj_threshold)

    data = speed[:, :, None]  # [T, N, 1]

    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    args.output_adj.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_data, data=data)
    np.save(args.output_adj, adj)

    density = float(np.count_nonzero(adj) / adj.size)
    print(f"Saved data to: {args.output_data} shape={data.shape} dtype={data.dtype}")
    print(
        f"Saved adj to: {args.output_adj} shape={adj.shape} nonzero={int(np.count_nonzero(adj))} density={density:.4f}"
    )


if __name__ == "__main__":
    main()

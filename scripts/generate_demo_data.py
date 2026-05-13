from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_adjacency(num_nodes: int, extra_edges: int = 2) -> np.ndarray:
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # Ring connections.
    for i in range(num_nodes):
        adj[i, (i - 1) % num_nodes] = 1.0
        adj[i, (i + 1) % num_nodes] = 1.0

    # Random shortcuts.
    rng = np.random.default_rng(42)
    for i in range(num_nodes):
        choices = rng.choice(num_nodes, size=extra_edges, replace=False)
        adj[i, choices] = 1.0

    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 1.0)
    return adj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--num_nodes", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=4000)
    parser.add_argument("--features", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    t = np.arange(args.timesteps, dtype=np.float32)
    adj = build_adjacency(args.num_nodes)

    daily = np.sin(2.0 * np.pi * t / 288.0)  # Simulate 5-min interval daily periodicity.
    weekly = np.sin(2.0 * np.pi * t / (288.0 * 7.0))
    base = 40.0 + 20.0 * daily + 8.0 * weekly

    node_scale = np.random.uniform(0.8, 1.2, size=(args.num_nodes,)).astype(np.float32)
    node_phase = np.random.uniform(-0.5, 0.5, size=(args.num_nodes,)).astype(np.float32)

    data = np.zeros((args.timesteps, args.num_nodes, args.features), dtype=np.float32)
    for n in range(args.num_nodes):
        shifted = np.roll(base, int(node_phase[n] * 20))
        noise = np.random.normal(0.0, 2.5, size=(args.timesteps,)).astype(np.float32)
        signal = shifted * node_scale[n] + noise
        signal = np.clip(signal, 0.0, None)
        data[:, n, 0] = signal

    # Introduce graph-correlated smoothing.
    row_sum = np.clip(adj.sum(axis=1, keepdims=True), 1e-8, None)
    adj_norm = adj / row_sum
    for ti in range(args.timesteps):
        data[ti, :, 0] = 0.65 * data[ti, :, 0] + 0.35 * (adj_norm @ data[ti, :, 0])

    np.savez_compressed(output_dir / "demo.npz", data=data)
    np.save(output_dir / "demo_adj.npy", adj)

    print(f"Saved demo data to: {output_dir / 'demo.npz'}")
    print(f"Saved demo adjacency to: {output_dir / 'demo_adj.npy'}")
    print(f"Data shape: {data.shape}")


if __name__ == "__main__":
    main()


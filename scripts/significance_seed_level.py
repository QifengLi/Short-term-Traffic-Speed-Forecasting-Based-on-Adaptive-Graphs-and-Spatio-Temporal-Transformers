from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


SEED_RE = re.compile(r"_seed(\d+)$")


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def extract_seed(run_name: str) -> int:
    match = SEED_RE.search(run_name)
    if match is None:
        raise ValueError(f"run_name does not end with _seed<id>: {run_name}")
    return int(match.group(1))


def exact_sign_flip_p_value(diff: np.ndarray) -> float:
    n = diff.shape[0]
    observed = abs(float(diff.mean()))
    total = 1 << n
    extreme = 0
    for mask in range(total):
        signs = np.ones(n, dtype=np.float64)
        for bit in range(n):
            if (mask >> bit) & 1:
                signs[bit] = -1.0
        stat = abs(float((diff * signs).mean()))
        if stat >= observed:
            extreme += 1
    return float((extreme + 1.0) / (total + 1.0))


def bootstrap_ci(diff: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = diff.shape[0]
    means = np.empty((n_boot,), dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(diff[idx].mean())
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def to_markdown(summary_rows: list[dict[str, Any]], per_seed_rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Seed-Level Significance Summary")
    lines.append("")
    lines.append("| Dataset | N Seeds | Mean(ref-ch) | Std(diff) | 95% CI | p-value | Ref Wins | Ch Wins | Significant(0.05) | Interpretation |")
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|---|---|")
    for row in summary_rows:
        ci_lo, ci_hi = row["bootstrap_95ci"]
        lines.append(
            f"| {row['dataset']} | {row['n_seeds']} | {row['mean_diff_ref_minus_challenger']:.6f} | "
            f"{row['std_diff_ref_minus_challenger']:.6f} | [{ci_lo:.6f}, {ci_hi:.6f}] | "
            f"{row['p_value_two_sided']:.6f} | {row['reference_win_count']} | {row['challenger_win_count']} | "
            f"{row['significant_at_0_05']} | {row['interpretation']} |"
        )
    lines.append("")
    lines.append("| Dataset | Seed | Ref MAE | Ch MAE | Ref-Ch | Better |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in per_seed_rows:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['reference_mae']:.6f} | {row['challenger_mae']:.6f} | "
            f"{row['diff_ref_minus_challenger']:.6f} | {row['better_model']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-metrics", nargs="+", required=True)
    parser.add_argument("--challenger-metrics", nargs="+", required=True)
    parser.add_argument("--reference-label", type=str, default="reference")
    parser.add_argument("--challenger-label", type=str, default="challenger")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=Path("outputs/improvements/seed_level_significance.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/improvements/seed_level_significance.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("outputs/improvements/seed_level_significance.md"))
    args = parser.parse_args()

    ref_map: dict[tuple[str, int], dict[str, Any]] = {}
    ch_map: dict[tuple[str, int], dict[str, Any]] = {}

    for p in [Path(x) for x in args.reference_metrics]:
        m = load_metrics(p)
        dataset = str(m["dataset"])
        run_name = str(m["run_name"])
        key = (dataset, extract_seed(run_name))
        ref_map[key] = m

    for p in [Path(x) for x in args.challenger_metrics]:
        m = load_metrics(p)
        dataset = str(m["dataset"])
        run_name = str(m["run_name"])
        key = (dataset, extract_seed(run_name))
        ch_map[key] = m

    common_keys = sorted(set(ref_map.keys()) & set(ch_map.keys()))
    if not common_keys:
        raise ValueError("No common (dataset, seed) pairs between reference and challenger metrics.")

    grouped: dict[str, list[tuple[int, dict[str, Any], dict[str, Any]]]] = {}
    for dataset, seed in common_keys:
        grouped.setdefault(dataset, []).append((seed, ref_map[(dataset, seed)], ch_map[(dataset, seed)]))

    summary_rows: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []

    for dataset, items in sorted(grouped.items()):
        items.sort(key=lambda x: x[0])
        seeds = [seed for seed, _, _ in items]
        ref_mae = np.array([float(ref["overall"]["mae"]) for _, ref, _ in items], dtype=np.float64)
        ch_mae = np.array([float(ch["overall"]["mae"]) for _, _, ch in items], dtype=np.float64)
        diff = ref_mae - ch_mae  # <0 means reference better

        p_value = exact_sign_flip_p_value(diff)
        ci_lo, ci_hi = bootstrap_ci(diff, n_boot=args.n_boot, seed=args.seed)
        ref_wins = int((diff < 0).sum())
        ch_wins = int((diff > 0).sum())
        ties = int((diff == 0).sum())

        if float(diff.mean()) < 0:
            interpretation = f"{args.reference_label} better"
        elif float(diff.mean()) > 0:
            interpretation = f"{args.challenger_label} better"
        else:
            interpretation = "tie"

        summary_rows.append(
            {
                "dataset": dataset,
                "reference_label": args.reference_label,
                "challenger_label": args.challenger_label,
                "n_seeds": len(seeds),
                "seeds": seeds,
                "mean_diff_ref_minus_challenger": float(diff.mean()),
                "std_diff_ref_minus_challenger": float(diff.std(ddof=0)),
                "bootstrap_95ci": [ci_lo, ci_hi],
                "p_value_two_sided": float(p_value),
                "reference_win_count": ref_wins,
                "challenger_win_count": ch_wins,
                "tie_count": ties,
                "significant_at_0_05": bool(p_value < 0.05),
                "interpretation": interpretation,
            }
        )

        for i, seed in enumerate(seeds):
            d = float(diff[i])
            better = args.reference_label if d < 0 else (args.challenger_label if d > 0 else "tie")
            per_seed_rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "reference_label": args.reference_label,
                    "challenger_label": args.challenger_label,
                    "reference_mae": float(ref_mae[i]),
                    "challenger_mae": float(ch_mae[i]),
                    "diff_ref_minus_challenger": d,
                    "better_model": better,
                }
            )

    payload = {
        "reference_label": args.reference_label,
        "challenger_label": args.challenger_label,
        "summary": summary_rows,
        "per_seed": per_seed_rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "seed",
                "reference_label",
                "challenger_label",
                "reference_mae",
                "challenger_mae",
                "diff_ref_minus_challenger",
                "better_model",
            ],
        )
        writer.writeheader()
        writer.writerows(per_seed_rows)

    args.out_md.write_text(to_markdown(summary_rows, per_seed_rows), encoding="utf-8")
    print(f"Saved JSON: {args.out_json}")
    print(f"Saved CSV:  {args.out_csv}")
    print(f"Saved MD:   {args.out_md}")


if __name__ == "__main__":
    main()


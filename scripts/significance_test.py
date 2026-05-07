from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


def load_pred(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    with np.load(path) as npz:
        preds = np.asarray(npz["preds"], dtype=np.float64)
        trues = np.asarray(npz["trues"], dtype=np.float64)
    return preds, trues


def parse_null_val(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null"}:
            return None
        if text == "nan":
            return float("nan")
        return float(text)
    raise ValueError(f"Unsupported null_val type: {type(value)}")


def load_null_val_from_config(config_path: Path) -> float | None:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_cfg = cfg.get("dataset", {})
    return parse_null_val(dataset_cfg.get("null_val", None))


def build_valid_mask(trues: np.ndarray, null_val: float | None) -> np.ndarray:
    if null_val is None:
        mask = np.ones_like(trues, dtype=bool)
    elif math.isnan(null_val):
        mask = ~np.isnan(trues)
    else:
        mask = trues != null_val
    return mask


def paired_permutation_test(diff: np.ndarray, num_perm: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    observed = np.abs(diff.mean())
    greater = 0
    for _ in range(num_perm):
        signs = rng.choice([-1.0, 1.0], size=diff.shape[0])
        permuted = np.abs((diff * signs).mean())
        if permuted >= observed:
            greater += 1
    return (greater + 1.0) / (num_perm + 1.0)


def bootstrap_ci(diff: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = diff.shape[0]
    means = np.empty((n_boot,), dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = diff[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def compare(
    reference: Path,
    challenger: Path,
    num_perm: int,
    n_boot: int,
    seed: int,
    null_val: float | None,
) -> dict:
    ref_pred, ref_true = load_pred(reference)
    ch_pred, ch_true = load_pred(challenger)
    if ref_pred.shape != ch_pred.shape or ref_true.shape != ch_true.shape or ref_true.shape != ref_pred.shape:
        raise ValueError("Shape mismatch among prediction files.")
    if not np.array_equal(ref_true, ch_true):
        raise ValueError("Ground-truth arrays mismatch between reference and challenger.")

    abs_ref = np.abs(ref_pred - ref_true)
    abs_ch = np.abs(ch_pred - ch_true)
    valid_mask = build_valid_mask(ref_true, null_val)
    if not np.any(valid_mask):
        raise ValueError("No valid points remain after applying null mask.")

    diff = (abs_ref - abs_ch)[valid_mask].reshape(-1)  # >0 means challenger better (lower error)

    p_value = paired_permutation_test(diff, num_perm=num_perm, seed=seed)
    ci_lo, ci_hi = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
    result = {
        "reference": str(reference),
        "challenger": str(challenger),
        "null_val": null_val,
        "n_valid_points": int(diff.shape[0]),
        "mean_error_diff_ref_minus_challenger": float(diff.mean()),
        "permutation_p_value_two_sided": float(p_value),
        "bootstrap_95ci": [ci_lo, ci_hi],
        "significant_at_0_05": bool(p_value < 0.05),
        "interpretation": ("challenger better" if diff.mean() > 0 else "reference better"),
    }
    return result


def to_markdown(results: list[dict]) -> str:
    lines = []
    lines.append("# 统计显著性检验报告")
    lines.append("")
    lines.append("| Reference | Challenger | Mean(ref-ch) | p-value | 95% CI | Significant(0.05) |")
    lines.append("|---|---|---:|---:|---|---|")
    for r in results:
        lines.append(
            f"| {Path(r['reference']).name} | {Path(r['challenger']).name} | "
            f"{r['mean_error_diff_ref_minus_challenger']:.6f} | {r['permutation_p_value_two_sided']:.6f} | "
            f"[{r['bootstrap_95ci'][0]:.6f}, {r['bootstrap_95ci'][1]:.6f}] | {r['significant_at_0_05']} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--challengers", nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--null-val",
        type=str,
        default=None,
        help="Override null value for masking. Use one of: numeric string, 'none', 'null', 'nan'.",
    )
    parser.add_argument("--num-perm", type=int, default=2000)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=Path("outputs/significance_test.json"))
    parser.add_argument("--out-md", type=Path, default=Path("outputs/significance_test.md"))
    args = parser.parse_args()

    if args.null_val is not None:
        null_val = parse_null_val(args.null_val)
    elif args.config is not None:
        null_val = load_null_val_from_config(args.config)
    else:
        null_val = None

    results = []
    for ch in args.challengers:
        results.append(
            compare(
                reference=args.reference,
                challenger=Path(ch),
                num_perm=args.num_perm,
                n_boot=args.n_boot,
                seed=args.seed,
                null_val=null_val,
            )
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text(to_markdown(results), encoding="utf-8")
    print(f"Saved JSON: {args.out_json}")
    print(f"Saved MD:   {args.out_md}")


if __name__ == "__main__":
    main()

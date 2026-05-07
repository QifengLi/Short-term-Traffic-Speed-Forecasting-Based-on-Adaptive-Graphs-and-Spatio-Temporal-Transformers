from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def make_output_paths(dataset_name: str, run_name: str, seed: int) -> tuple[str, str]:
    pred = f"outputs/robustness/{dataset_name}_{run_name}_seed{seed}_predictions.npz"
    metrics = f"outputs/robustness/{dataset_name}_{run_name}_seed{seed}_metrics.json"
    return pred, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 2026])
    parser.add_argument("--out-dir", type=Path, default=Path("configs/seed_sweeps"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for cfg_path in [Path(p) for p in args.base_configs]:
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        dataset_name = str(cfg["dataset"]["name"])
        base_run_name = str(cfg.get("experiment", {}).get("name", cfg["model"]["name"])).lower()

        for seed in args.seeds:
            new_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            new_run_name = f"{base_run_name}_seed{seed}"
            new_cfg.setdefault("training", {})
            new_cfg["training"]["seed"] = int(seed)
            new_cfg.setdefault("experiment", {})
            new_cfg["experiment"]["name"] = new_run_name

            pred_path, metrics_path = make_output_paths(dataset_name, base_run_name, int(seed))
            new_cfg.setdefault("output", {})
            new_cfg["output"]["prediction_path"] = pred_path
            new_cfg["output"]["metrics_path"] = metrics_path
            new_cfg["output"]["save_predictions"] = True
            new_cfg["output"]["save_metrics"] = True

            out_path = args.out_dir / f"{cfg_path.stem}_seed{seed}.yaml"
            out_path.write_text(yaml.safe_dump(new_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
            generated.append(out_path)

    print("Generated configs:")
    for p in generated:
        print(str(p).replace("\\", "/"))


if __name__ == "__main__":
    main()

# src/ssl_for_ood/training/orchestrator.py

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ssl_for_ood.training.trainer import load_config, train_single_run
from ssl_for_ood.training.search import sample_from_space, load_yaml

def run_experiments(
    project_root: Path,
    model_names: list[str],
    dataset_names: list[str],
    n_trials: int = 20,
    use_mlflow: bool = True,
) -> pd.DataFrame:
    
    results = []

    for model_name in model_names:
        for dataset_name in dataset_names:

            print(f"\n🚀 Running {model_name} on {dataset_name}")

            base_config_path = project_root / "config" / model_name / "base.yaml"
            dataset_config_path = project_root / "config" / model_name / f"{dataset_name}.yaml"
            search_space_path = project_root / "config" / model_name / "search_space.yaml"

            base_config = load_config(base_config_path, dataset_config_path)
            search_space = load_yaml(search_space_path)

            for trial in range(n_trials):
                print(f"[{model_name} | {dataset_name}] Trial {trial+1}/{n_trials}")

                sampled_config = sample_from_space(search_space)
                config = merge_config(base_config, sampled_config)

                config["seed"] = 42 + trial
                config["mlflow"]["enabled"] = use_mlflow
                config["mlflow"]["run_name"] = f"{model_name}__{dataset_name}__trial_{trial}"
                config["output"]["run_name"] = f"{model_name}_{dataset_name}_trial_{trial}"

                start_time = time.time()

                try:
                    metrics, artifacts = train_single_run(config)

                    duration = time.time() - start_time

                    results.append({
                        "model": model_name,
                        "dataset": dataset_name,
                        "trial": trial,
                        "duration_sec": duration,
                        "config": json.dumps(sampled_config),

                        "val_id_mae": metrics["val_id"]["mae"],
                        "val_ood_mae": metrics["val_ood"]["mae"],
                        "val_id_std": metrics["val_id"]["std_abs_error"],
                        "val_ood_std": metrics["val_ood"]["std_abs_error"],
                        "selected_metric": metrics["selected_metric"],

                        "best_epoch": metrics.get("best_epoch"),
                        "test_mae": metrics.get("test", {}).get("mae"),

                        "run_dir": str(artifacts.run_dir),
                    })

                except Exception as e:
                    print(f"❌ Trial failed: {e}")
                    continue

    df = pd.DataFrame(results)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = project_root / "results" / f"hyperparameter_search_{timestamp}.csv"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(out_path, index=False)

    # best configs
    best_df = df.loc[df.groupby(["model", "dataset"])["selected_metric"].idxmin()]
    best_df.to_csv(project_root / "results" / "best_configs.csv", index=False)

    print(f"\n✅ Saved results to {out_path}")

    return df


def merge_config(base: dict[str, Any], sampled: dict[str, Any]) -> dict[str, Any]:
    from copy import deepcopy

    def _merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                _merge(a[k], v)
            else:
                a[k] = v
        return a

    return _merge(deepcopy(base), sampled)

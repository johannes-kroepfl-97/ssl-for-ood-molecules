from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from ssl_for_ood.training.trainer import load_config, train_single_run


NORMALIZATION_STRATEGIES = [
    None,
    "architecture_native_norm",
    "input_bn",
    "final_bn",
]


def set_normalization_strategy(
    config: dict[str, Any],
    strategy: str | None,
) -> dict[str, Any]:
    config = deepcopy(config)
    config.setdefault("model", {})
    config["model"]["normalization_strategy"] = strategy
    return config


def run_normalization_experiments(
    project_root: Path,
    model_names: list[str],
    dataset_names: list[str],
    strategies: list[str | None] | None = None,
    use_mlflow: bool = True,
) -> pd.DataFrame:
    strategies = strategies or NORMALIZATION_STRATEGIES

    rows = []

    for dataset_name in dataset_names:
        for model_name in model_names:
            base_config_path = project_root / "config" / model_name / "base.yaml"
            dataset_config_path = project_root / "config" / model_name / f"{dataset_name}.yaml"

            base_config = load_config(base_config_path, dataset_config_path)

            for strategy in strategies:
                strategy_name = "baseline" if strategy is None else strategy

                print(f"\n🚀 {dataset_name} | {model_name} | {strategy_name}")

                config = set_normalization_strategy(base_config, strategy)

                config["model_name"] = model_name
                config.setdefault("dataset", {})
                config["dataset"]["name"] = dataset_name

                config.setdefault("training", {})
                config["training"]["evaluate_test"] = True

                config.setdefault("mlflow", {})
                config["mlflow"]["enabled"] = use_mlflow
                config["mlflow"]["run_name"] = f"{model_name}__{dataset_name}__{strategy_name}"

                config.setdefault("output", {})
                config["output"]["run_name"] = f"{model_name}_{dataset_name}_{strategy_name}"

                start = time.time()

                try:
                    metrics, artifacts = train_single_run(config)
                    duration = time.time() - start

                    rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "normalization_strategy": strategy,
                            "strategy_name": strategy_name,
                            "duration_sec": duration,
                            "val_id_mae": metrics["val_id"]["mae"],
                            "val_ood_mae": metrics["val_ood"]["mae"],
                            "test_mae": metrics.get("test", {}).get("mae"),
                            "val_id_std_abs_error": metrics["val_id"]["std_abs_error"],
                            "val_ood_std_abs_error": metrics["val_ood"]["std_abs_error"],
                            "test_std_abs_error": metrics.get("test", {}).get("std_abs_error"),
                            "run_dir": str(artifacts.run_dir),
                            "config": json.dumps(config),
                        }
                    )

                except Exception as exc:
                    print(f"❌ Failed: {dataset_name} | {model_name} | {strategy_name}: {exc}")
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "normalization_strategy": strategy,
                            "strategy_name": strategy_name,
                            "error": str(exc),
                        }
                    )

    df = pd.DataFrame(rows)

    out_dir = project_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"normalization_experiments_{timestamp}.csv"
    df.to_csv(out_path, index=False)

    print(f"\n✅ Saved normalization experiment results to {out_path}")

    return df

def run_normalization_from_best_configs(
    project_root,
    search_results_path,
    dataset_name="aav",
    strategies=None,
):
    import json
    import time
    from pathlib import Path
    import pandas as pd

    from ssl_for_ood.training.trainer import load_config, train_single_run

    strategies = strategies or [
        "architecture_native_norm",
        "input_bn",
        "final_bn",
        None,
    ]

    df_old = pd.read_csv(search_results_path)

    # --- Select best config per model ---
    # prefer val_ood_mae if available
    metric_col = "val_ood_mae" if "val_ood_mae" in df_old.columns else "val_id_mae"

    model_col = "model" if "model" in df_old.columns else "model_name"

    best_rows = (
        df_old
        .sort_values(metric_col)
        .groupby(model_col, as_index=False)
        .first()
    )

    results = []

    for _, row in best_rows.iterrows():

        model_name = row[model_col]

        print(f"""
###############
### {model_name}
###############
""")

        # load base + dataset config
        config = load_config(
            project_root / "config" / model_name / "base.yaml",
            project_root / "config" / model_name / f"{dataset_name}.yaml",
        )

        # --- Inject best hyperparameters ---
        if "config" in row and pd.notna(row["config"]):
            old_config = json.loads(row["config"])

            config["model"].update(old_config.get("model", {}))
            config["training"].update(old_config.get("training", {}))

        else:
            # fallback: match keys manually
            for key in row.index:
                if key in config["model"]:
                    config["model"][key] = row[key]
                if key in config["training"]:
                    config["training"][key] = row[key]

        for strategy in strategies:

            print(f'   --> STRATEGY: "{strategy}"')

            # deepcopy via json (safe for configs)
            config_run = json.loads(json.dumps(config))

            config_run["model"]["normalization_strategy"] = strategy
            config_run["dataset"]["name"] = dataset_name
            
            # config_run["training"]["epochs"] = 2

            start = time.time()

            try:
                metrics, artifacts = train_single_run(config_run)
                duration = time.time() - start

                results.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "normalization_strategy": strategy,
                    "duration_sec": duration,
                    "val_id_mae": metrics["val_id"]["mae"],
                    "val_ood_mae": metrics["val_ood"]["mae"],
                    "test_mae": metrics.get("test", {}).get("mae"),
                    "val_id_std_abs_error": metrics["val_id"]["std_abs_error"],
                    "val_ood_std_abs_error": metrics["val_ood"]["std_abs_error"],
                    "test_std_abs_error": metrics.get("test", {}).get("std_abs_error"),
                    "run_dir": str(artifacts.run_dir),
                })

            except Exception as e:
                print(f"❌ Failed: {model_name} | {strategy}: {e}")

                results.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "normalization_strategy": strategy,
                    "error": str(e),
                })

    df_results = pd.DataFrame(results)

    # --- Save results (same style as your search) ---
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"normalization_from_best_{dataset_name}_{timestamp}.csv"

    df_results.to_csv(out_path, index=False)

    print(f"\n✅ Saved normalization results to {out_path}")

    return df_results
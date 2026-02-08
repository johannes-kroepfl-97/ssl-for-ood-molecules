"""
io.py

Dataset loaders that:
- resolve the project-root data directory robustly (independent of CWD)
- build mutation-distance-based domain shift splits (ID train, ID val for early stopping,
  near-OOD val for hyperparameter selection, target_unlabeled for SSL/DA, far-OOD test)
- save CSVs + metadata.txt per dataset
- expose no-input loader functions with consistent return signatures
"""

from datasets import load_dataset
import pandas as pd
import os
from pathlib import Path


# -----------------------------
# Data root resolution (clean + robust)
# -----------------------------

def _find_repo_root(start: Path) -> Path:
    """
    Walk upward from a starting path until we find repo markers.
    Prefers pyproject.toml, falls back to .git.
    """
    start = start.resolve()
    for p in (start, *start.parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    # Fallback: assumes src layout and goes up 3 from io.py (repo/src/pkg/data/io.py)
    return start.parents[3]


def _get_data_root() -> Path:
    """
    Resolve base data directory.
    Order:
      1) env var SSL_FOR_OOD_DATA_DIR
      2) <repo_root>/data
    """
    env = os.getenv("SSL_FOR_OOD_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (_find_repo_root(Path(__file__)) / "data").resolve()


DATA_ROOT = _get_data_root()


def _dataset_splits_dir(dataset_name: str) -> Path:
    d = DATA_ROOT / dataset_name / "splits"
    d.mkdir(parents=True, exist_ok=True)
    return d


# -----------------------------
# shared helpers (AAV-style, minimal)
# -----------------------------

def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def _save_files_with_metadata_txt(
    out_dir: Path,
    *,
    files: dict,          # filename -> df
    meta_lines: list,
):
    """
    Writes arbitrary CSV files + metadata.txt into out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname, df in files.items():
        df.to_csv(out_dir / fname, index=False)

    meta_path = out_dir / "metadata.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        for line in meta_lines:
            f.write(line.rstrip("\n") + "\n")


def _consensus_anchor_fixed(seqs):
    """
    Majority vote per position (assumes fixed length).
    """
    from collections import Counter

    L = len(seqs[0])
    cons = []
    for i in range(L):
        cons.append(Counter(s[i] for s in seqs).most_common(1)[0][0])
    return "".join(cons)


def _filter_to_dominant_length(df: pd.DataFrame, seq_col: str = "sequence"):
    """
    Ensures fixed length for Hamming distance by filtering to dominant length.
    Returns (df_filtered, L_mode).
    """
    seqs = df[seq_col].astype(str)
    lens = seqs.apply(len)
    L_mode = int(lens.mode().iloc[0])
    out = df[lens == L_mode].copy().reset_index(drop=True)
    return out, L_mode


def _add_mutation_distance(df: pd.DataFrame, *, core: str, seq_col: str = "sequence"):
    out = df.copy()
    out[seq_col] = out[seq_col].astype(str)
    out["mut_dist"] = out[seq_col].apply(lambda s: _hamming(s, core))
    return out


def _finalize_schema(df: pd.DataFrame, *, split_name: str):
    """
    Final column order:
      sequence,label,mut_dist,split
    """
    out = df.copy()
    out["sequence"] = out["sequence"].astype(str)
    out["label"] = out["label"].astype(float)
    out["mut_dist"] = out["mut_dist"].astype(int)
    out["split"] = split_name
    return out[["sequence", "label", "mut_dist", "split"]].reset_index(drop=True)


def _sample_df(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Deterministic sample without replacement (or all if df smaller than n).
    """
    if n <= 0:
        return df.iloc[0:0].copy()
    if len(df) <= n:
        return df.copy().reset_index(drop=True)
    return df.sample(n=n, replace=False, random_state=seed).reset_index(drop=True)


def _split_id_train_val_id(id_df: pd.DataFrame, val_id_frac: float = 0.10, seed: int = 42):
    """
    Create val_id from ID pool (for early stopping), and the remaining ID pool is train.
    """
    val_id_df = id_df.sample(frac=val_id_frac, random_state=seed).copy()
    train_df = id_df.drop(index=val_id_df.index).copy()
    return train_df.reset_index(drop=True), val_id_df.reset_index(drop=True)


def _quantile_thresholds_for_shift(
    dists: pd.Series,
    *,
    q_val_ood: float = 0.70,
    q_test: float = 0.90,
):
    """
    Defines disjoint bands:
      train ID: mut_dist < thr_val
      val_ood:  thr_val <= mut_dist < thr_test
      test:     mut_dist >= thr_test
    """
    import numpy as np

    d = dists.to_numpy()
    thr_val = float(np.quantile(d, q_val_ood))
    thr_test = float(np.quantile(d, q_test))

    if thr_test < thr_val:
        thr_test = thr_val

    return thr_val, thr_test


# Naming: common in domain adaptation literature
TARGET_UNLABELED_FNAME = "target_unlabeled.csv"


# -----------------------------
# GFP
# -----------------------------

def load_gfp_data():
    """
    Produces:
      train.csv            (ID train)
      val_id.csv           (ID validation for early stopping)
      val_ood.csv          (near-OOD validation for hyperparameter selection)
      target_unlabeled.csv (remaining near-OOD pool not used in val_ood; for SSL/DA; ignore labels)
      test.csv             (far-OOD test)
      metadata.txt

    Returns:
      (train_df, val_ood_df, test_df, out_dir)
    """
    repo_id = "InstaDeepAI/true-cds-protein-tasks"
    subset = "fluorescence"
    ds = load_dataset(repo_id, name=subset)

    # Merge all splits, then re-split by mut_dist bands
    all_seqs = []
    all_labels = []
    for split_name in ds.keys():
        all_seqs.extend([str(s) for s in ds[split_name]["sequence"]])
        all_labels.extend([float(y) for y in ds[split_name]["label"]])

    df = pd.DataFrame({"sequence": all_seqs, "label": all_labels})

    # Fixed length for Hamming
    df, L_mode = _filter_to_dominant_length(df, seq_col="sequence")

    # Core: consensus on combined pool
    core = _consensus_anchor_fixed(df["sequence"].tolist())

    # Distances
    df = _add_mutation_distance(df, core=core, seq_col="sequence")

    # Quantile bands (defaults):
    # val_ood starts at 70th percentile, test at 90th percentile
    thr_val, thr_test = _quantile_thresholds_for_shift(df["mut_dist"], q_val_ood=0.70, q_test=0.90)

    id_pool = df[df["mut_dist"] < thr_val].copy().reset_index(drop=True)
    val_ood_pool = df[(df["mut_dist"] >= thr_val) & (df["mut_dist"] < thr_test)].copy().reset_index(drop=True)
    test_pool = df[df["mut_dist"] >= thr_test].copy().reset_index(drop=True)

    # val_id from ID pool (10%) and remaining is train
    train_id_pool, val_id_df_raw = _split_id_train_val_id(id_pool, val_id_frac=0.10, seed=42)

    # cap near-OOD val to 5000 for predictable tuning cost
    val_ood_df_raw = _sample_df(val_ood_pool, n=5000, seed=42)

    # target_unlabeled: remaining near-OOD pool not used in val_ood (for DA/SSL)
    target_unlabeled_raw = val_ood_pool.drop(index=val_ood_df_raw.index, errors="ignore").copy().reset_index(drop=True)

    train_df = _finalize_schema(train_id_pool, split_name="train")
    val_id_df = _finalize_schema(val_id_df_raw, split_name="val_id")
    val_ood_df = _finalize_schema(val_ood_df_raw, split_name="val_ood")
    target_unlabeled_df = _finalize_schema(target_unlabeled_raw, split_name="target_unlabeled")
    test_df = _finalize_schema(test_pool, split_name="test")

    out_dir = _dataset_splits_dir("gfp")

    meta_lines = [
        "dataset=gfp",
        f"repo_id={repo_id}",
        f"subset={subset}",
        f"hf_splits_present={list(ds.keys())}",
        f"data_root={DATA_ROOT}",
        f"out_dir={out_dir}",
        f"dominant_length_L_mode={L_mode}",
        "core_definition=consensus_on_combined_filtered_pool",
        f"core_length={len(core)}",
        "split_strategy=disjoint_mut_dist_bands",
        "band_definitions=train: mut_dist < thr_val; val_ood: thr_val <= mut_dist < thr_test; test: mut_dist >= thr_test",
        "val_id_definition=10% random sample from train band (ID) for early stopping",
        "q_val_ood=0.70",
        "q_test=0.90",
        f"thr_val={thr_val}",
        f"thr_test={thr_test}",
        "val_ood_cap_n=5000",
        "split_seed=42",
        f"counts_total_filtered={len(df)}",
        f"counts_train={len(train_df)}",
        f"counts_val_id={len(val_id_df)}",
        f"counts_val_ood={len(val_ood_df)}",
        f"counts_target_unlabeled={len(target_unlabeled_df)}",
        f"counts_test={len(test_df)}",
    ]

    _save_files_with_metadata_txt(
        out_dir,
        files={
            "train.csv": train_df,
            "val_id.csv": val_id_df,
            "val_ood.csv": val_ood_df,
            TARGET_UNLABELED_FNAME: target_unlabeled_df,
            "test.csv": test_df,
        },
        meta_lines=meta_lines,
    )

    return train_df, val_ood_df, test_df, str(out_dir)


# -----------------------------
# AAV
# -----------------------------

def load_aav_data():
    """
    Same outputs as GFP:
      train.csv, val_id.csv, val_ood.csv, target_unlabeled.csv, test.csv, metadata.txt

    Returns:
      (train_df, val_ood_df, test_df, out_dir)
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    repo_id = "AI4Protein/FLIP_AAV_des-mut"

    def _download_split_df(split_name: str):
        filename = f"{split_name}.csv"
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        return pd.read_csv(path), path

    train_raw, train_path = _download_split_df("train")
    valid_raw, valid_path = _download_split_df("valid")
    test_raw, test_path = _download_split_df("test")

    # Merge official splits, then re-split by mut_dist bands
    df = pd.concat([train_raw, valid_raw, test_raw], axis=0, ignore_index=True)

    # Normalize columns
    df["sequence"] = df["aa_seq"].astype(str)
    df["label"] = df["label"].astype(float)
    df = df[["sequence", "label"]]

    # Fixed length for Hamming
    df, L_mode = _filter_to_dominant_length(df, seq_col="sequence")

    # Core: consensus on combined pool
    core = _consensus_anchor_fixed(df["sequence"].tolist())

    # Distances
    df = _add_mutation_distance(df, core=core, seq_col="sequence")

    # AAV is large; use slightly stricter banding:
    # val_ood at 75th percentile, test at 90th percentile
    thr_val, thr_test = _quantile_thresholds_for_shift(df["mut_dist"], q_val_ood=0.75, q_test=0.90)

    id_pool = df[df["mut_dist"] < thr_val].copy().reset_index(drop=True)
    val_ood_pool = df[(df["mut_dist"] >= thr_val) & (df["mut_dist"] < thr_test)].copy().reset_index(drop=True)
    test_pool = df[df["mut_dist"] >= thr_test].copy().reset_index(drop=True)

    train_id_pool, val_id_df_raw = _split_id_train_val_id(id_pool, val_id_frac=0.10, seed=42)

    # cap near-OOD val to 5000
    val_ood_df_raw = _sample_df(val_ood_pool, n=5000, seed=42)

    target_unlabeled_raw = val_ood_pool.drop(index=val_ood_df_raw.index, errors="ignore").copy().reset_index(drop=True)

    train_df = _finalize_schema(train_id_pool, split_name="train")
    val_id_df = _finalize_schema(val_id_df_raw, split_name="val_id")
    val_ood_df = _finalize_schema(val_ood_df_raw, split_name="val_ood")
    target_unlabeled_df = _finalize_schema(target_unlabeled_raw, split_name="target_unlabeled")
    test_df = _finalize_schema(test_pool, split_name="test")

    out_dir = _dataset_splits_dir("aav")

    meta_lines = [
        "dataset=aav",
        f"repo_id={repo_id}",
        f"train_source={train_path}",
        f"valid_source={valid_path}",
        f"test_source={test_path}",
        f"data_root={DATA_ROOT}",
        f"out_dir={out_dir}",
        f"dominant_length_L_mode={L_mode}",
        "core_definition=consensus_on_combined_filtered_pool",
        f"core_length={len(core)}",
        "split_strategy=disjoint_mut_dist_bands",
        "band_definitions=train: mut_dist < thr_val; val_ood: thr_val <= mut_dist < thr_test; test: mut_dist >= thr_test",
        "val_id_definition=10% random sample from train band (ID) for early stopping",
        "q_val_ood=0.75",
        "q_test=0.90",
        f"thr_val={thr_val}",
        f"thr_test={thr_test}",
        "val_ood_cap_n=5000",
        "split_seed=42",
        f"counts_total_filtered={len(df)}",
        f"counts_train={len(train_df)}",
        f"counts_val_id={len(val_id_df)}",
        f"counts_val_ood={len(val_ood_df)}",
        f"counts_target_unlabeled={len(target_unlabeled_df)}",
        f"counts_test={len(test_df)}",
    ]

    _save_files_with_metadata_txt(
        out_dir,
        files={
            "train.csv": train_df,
            "val_id.csv": val_id_df,
            "val_ood.csv": val_ood_df,
            TARGET_UNLABELED_FNAME: target_unlabeled_df,
            "test.csv": test_df,
        },
        meta_lines=meta_lines,
    )

    return train_df, val_ood_df, test_df, str(out_dir)


# -----------------------------
# TFBind8
# -----------------------------

def load_tfbind8_data():
    """
    TFBind8 template:
      - core: random anchor (seed=42)
      - train band: mut_dist in {1,2,3,4,5}
      - val_ood band: mut_dist == 6 (sample 5000)
      - test band: mut_dist in {7,8}
      - val_id: 10% random sample from train band (ID) for early stopping
      - target_unlabeled: remaining mut_dist==6 not used in val_ood

    Outputs:
      train.csv, val_id.csv, val_ood.csv, target_unlabeled.csv, test.csv, metadata.txt

    Returns:
      (train_df, val_ood_df, test_df, out_dir)
    """
    import numpy as np
    from huggingface_hub import hf_hub_download

    repo_id = "beckhamc/design_bench_data"
    x_file = "tf_bind_8-SIX6_REF_R1/tf_bind_8-x-0.npy"
    y_file = "tf_bind_8-SIX6_REF_R1/tf_bind_8-y-0.npy"

    def _x_to_dna_strings(x: np.ndarray, alphabet=("A", "C", "G", "T")):
        x = np.asarray(x)

        if x.ndim == 2:
            if x.max() > 3 or x.min() < 0:
                raise ValueError(f"Expected integer tokens in [0,3], got min={x.min()} max={x.max()}")
            idx = x.astype(int)
        elif x.ndim == 3:
            if x.shape[-1] == 4:
                idx = x.argmax(axis=-1)
            elif x.shape[1] == 4:
                idx = x.argmax(axis=1)
            else:
                raise ValueError(f"Unrecognized one-hot shape {x.shape}; expected (...,4) somewhere.")
        else:
            raise ValueError(f"Unsupported x ndim={x.ndim}; shape={x.shape}")

        alpha = np.array(alphabet)
        return ["".join(alpha[row]) for row in idx]

    # Download
    x_path = hf_hub_download(repo_id=repo_id, filename=x_file, repo_type="dataset")
    y_path = hf_hub_download(repo_id=repo_id, filename=y_file, repo_type="dataset")

    x_raw = np.load(x_path, allow_pickle=False)
    y_raw = np.load(y_path, allow_pickle=False).reshape(-1).astype(float)

    sequences = _x_to_dna_strings(x_raw, alphabet=("A", "C", "G", "T"))
    if not sequences:
        raise ValueError("Empty TFBind8 dataset.")
    seq_len = len(sequences[0])
    if seq_len != 8:
        raise ValueError(f"Expected TFBind8 sequence length 8, got seq_len={seq_len}")

    # Core: random anchor (deterministic)
    np.random.seed(42)
    anchor_idx = int(np.random.choice(len(sequences)))
    core = sequences[anchor_idx]

    df = pd.DataFrame({"sequence": pd.Series(sequences, dtype="string"), "label": y_raw.astype(float)})
    df = _add_mutation_distance(df, core=core, seq_col="sequence")

    # Bands
    train_band = df[df["mut_dist"].isin([1, 2, 3, 4, 5])].copy().reset_index(drop=True)
    val_ood_band = df[df["mut_dist"] == 6].copy().reset_index(drop=True)
    test_band = df[df["mut_dist"].isin([7, 8])].copy().reset_index(drop=True)

    # val_id for early stopping from train band
    train_id_pool, val_id_df_raw = _split_id_train_val_id(train_band, val_id_frac=0.10, seed=42)

    # val_ood: sample 5000
    val_ood_df_raw = _sample_df(val_ood_band, n=5000, seed=42)

    # target_unlabeled: remaining from dist==6
    target_unlabeled_raw = val_ood_band.drop(index=val_ood_df_raw.index, errors="ignore").copy().reset_index(drop=True)

    train_df = _finalize_schema(train_id_pool, split_name="train")
    val_id_df = _finalize_schema(val_id_df_raw, split_name="val_id")
    val_ood_df = _finalize_schema(val_ood_df_raw, split_name="val_ood")
    target_unlabeled_df = _finalize_schema(target_unlabeled_raw, split_name="target_unlabeled")
    test_df = _finalize_schema(test_band, split_name="test")

    out_dir = _dataset_splits_dir("tfbind8")

    meta_lines = [
        "dataset=tfbind8",
        f"repo_id={repo_id}",
        f"x_source={x_path}",
        f"y_source={y_path}",
        f"data_root={DATA_ROOT}",
        f"out_dir={out_dir}",
        f"seq_len={seq_len}",
        "core_definition=random_anchor_sequence",
        "anchor_seed=42",
        f"anchor_idx={anchor_idx}",
        f"core_anchor_seq={core}",
        "split_strategy=disjoint_distance_bands_fixed",
        "train_definition=mut_dist in {1,2,3,4,5}",
        "val_id_definition=10% random sample from train band for early stopping",
        "val_ood_definition=mut_dist==6 (sample 5000) for hyperparameter selection",
        "target_unlabeled_definition=remaining mut_dist==6 not used in val_ood (for SSL/DA; ignore labels)",
        "test_definition=mut_dist in {7,8}",
        "val_ood_cap_n=5000",
        "split_seed=42",
        f"counts_total={len(df)}",
        f"counts_train={len(train_df)}",
        f"counts_val_id={len(val_id_df)}",
        f"counts_val_ood={len(val_ood_df)}",
        f"counts_target_unlabeled={len(target_unlabeled_df)}",
        f"counts_test={len(test_df)}",
    ]

    _save_files_with_metadata_txt(
        out_dir,
        files={
            "train.csv": train_df,
            "val_id.csv": val_id_df,
            "val_ood.csv": val_ood_df,
            TARGET_UNLABELED_FNAME: target_unlabeled_df,
            "test.csv": test_df,
        },
        meta_lines=meta_lines,
    )

    return train_df, val_ood_df, test_df, str(out_dir)


# -----------------------------
# Convenience
# -----------------------------

def load_all_data():
    """
    Convenience loader. Returns a dict of dataset_name -> (train_df, val_ood_df, test_df, out_dir).
    """
    out = {}
    out["gfp"] = load_gfp_data()
    out["aav"] = load_aav_data()
    out["tfbind8"] = load_tfbind8_data()
    return out

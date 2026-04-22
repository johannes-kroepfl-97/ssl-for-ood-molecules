# SSL for OOD Molecule Data

In this repository we explore evaluate self-supervised learning methods for measuring out-of-distribution domain shift in 3 molecular datasets.

## Data

### Datasets

We use three sequence–function datasets that differ in modality, sequence length, and mutation structure:

- GFP – Protein fluorescence prediction (taken from https://huggingface.co/datasets/OATML-Markslab/ProteinGym_v1/tree/main/DMS_substitutions where we selected for DMS_id=='GFP_AEQVI_Sarkisyan_2016')
- AAV – AAV capsid fitness landscape (taken from https://huggingface.co/datasets/OATML-Markslab/ProteinGym_v1/tree/main/DMS_substitutions where we selected for DMS_id=='CAPSD_AAV2S_Sinai_2021')
- TFBind8 – DNA–protein binding (TFBind8, SIX6_REF_R1; Design-Bench)

All datasets are converted to a common CSV schema:
sequence (string), label (float), mut_dist (mutation distance), split.

All mutation distances are computed by using a wild type. The wild types for GFP and AAV are provided by protein gym in the column "target_seq". The wild type for TFBind8 got selected randomly.

### Split Strategy (ID → near-OOD → far-OOD)

Our goal is to study out-of-distribution (OOD) generalization and domain adaptation under a controlled and interpretable shift. For all datasets, we define a core (wild-type or anchor sequence) and compute Hamming distance (mut_dist) to that core.

We then create disjoint splits with increasing distribution shift:

- Train (ID): Low mutation distance; used for supervised training.
- Validation (ID) (val_id.csv): 10% random holdout from the train distribution; used only for early stopping and training stability.
- Validation (near-OOD) (val_ood.csv): Moderately shifted sequences; used for hyperparameter selection, encouraging robustness rather than pure ID performance. Capped to a fixed size (5,000) for stable and comparable tuning.
- Target Unlabeled (target_unlabeled.csv): Additional near-OOD data not used for validation; intended for self-supervised learning / domain adaptation (labels present but ignored during training).
- Test (far-OOD): Highest mutation distances; used only for final evaluation.

For TFBind8, mutation-distance bands are fixed (train: 1–5, val_ood: 6, test: 7–8).
For GFP and AAV, bands are defined via dataset-specific mutation-distance quantiles to ensure meaningful shift while retaining sufficient sample sizes.

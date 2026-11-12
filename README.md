# CrossMamba-DTA

**A Cross-Modal State Space Model for Drug–Target Affinity Prediction** · Wei Lan, Tian Huang, Qingfeng Chen, Min Li, Yi Pan · 

CrossMamba-DTA predicts binding affinity from a SMILES string and a protein sequence. ChemBERTa (768-d) and ESM-2 (1280-d) embeddings are projected into a shared latent space by a **LocalConvAdapter**, mixed on a **joint drug–target sequence** by **bidirectional Mamba** scans with **gated residual fusion**, and then aggregated by **masked GeM pooling** that discards padding and emphasises strong local responses. Cross-modal cost is linear in sequence length.

```
SMILES  → ChemBERTa → LocalConvAdapter ┐
                                       ├→ Gated Cross-Mamba → Masked GeM → [h_d ; h_p] → MLP → ŷ
Protein → ESM-2     → LocalConvAdapter ┘
```

| Dataset | MSE ↓     | CI ↑      | r²m ↑     | Best baseline (MSE) |
| :------ | :-------- | :-------- | :-------- | :------------------ |
| Davis   | **0.174** | **0.900** | **0.728** | 0.216 AttentionDTA  |
| KIBA    | **0.142** | 0.889     | **0.790** | 0.150 MATT-DTI      |

5-fold cross-validation, 7:1:2 train/validation/test split. Ablation on Davis (full model MSE 0.174): w/o GeM 0.189, w/o padding mask 0.186, w/o gated Cross-Mamba 0.178, w/o LocalConvAdapter 0.178. Best configuration: hidden dimension 512, d_state 8.

## Quick start

```bash
pip install torch causal-conv1d mamba-ssm transformers rdkit scikit-learn pandas numpy matplotlib openpyxl tqdm
python row_data_division.py        # 5-fold splits → dataset/splits_dataset/
python run_processing.py           # ChemBERTa + ESM-2 embeddings → processed_data/
python train.py                    # 5-fold training → kiba_best_model_fold_{0..4}.pt
python plot_predicted_vs_true.py   # predicted-vs-observed scatter + per-prediction CSV
```

Data: `dataset/Davis.csv` and `dataset/KIBA.csv` with columns `COMPOUND_SMILES`, `PROTEIN_SEQUENCE`, `REG_LABEL` (pKd). An NVIDIA GPU is required, since `mamba-ssm` is CUDA-only. Defaults: hidden 512, d_state 8, AdamW (lr 3e-5, wd 1e-3), cosine decay, batch 256, MSE on z-scored labels, early stopping (patience 60), fp16 AMP. Sequences are truncated at 128 tokens (drug) and 1024 residues (protein). Test metrics are reported after de-normalisation.

## Code map

`model.py`, `model_utils.py` — adapters, bidirectional Mamba, gates, GeM pooling, MLP head. `dataloader.py` — feature cache, padding and masks. `metrics.py` — CI and r²m. `train.py` — 5-fold CV loop. `row_data_division.py` → `run_processing.py` → `train.py` — full pipeline. `plot_predicted_vs_true.py` — prediction scatter. `case_analysis.py` — external glaucoma-related pairs; `functional_group_perturbation_analysis.py` and `target_region_perturbation_analysis.py` — RDKit functional-group and UniProt region masking, averaged over folds.

Notes: checkpoints are always written as `kiba_best_model_fold_{fold}.pt` regardless of the dataset; the perturbation scripts read `case_analysis_results/low_error_cases.csv` and `all_case_predictions.csv`, exported from the case-analysis output. Padding masks use `True` for padded positions. Weight files, processed features and datasets are not included in this repository.

Cite: W. Lan, T. Huang, Q. Chen, M. Li, Y. Pan. CrossMamba-DTA: A Cross-Modal State Space Model for Drug-Target Affinity Prediction. *Bioinformatics*, 2026.

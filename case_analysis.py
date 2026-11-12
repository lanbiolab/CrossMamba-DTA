import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


DEFAULT_CODE_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_FILE = DEFAULT_CODE_DIR / "case_change.xlsx"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Case analysis for drug-target affinity prediction."
    )
    parser.add_argument("--code_dir", type=Path, default=DEFAULT_CODE_DIR)
    parser.add_argument("--case_file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument("--dataset_name", type=str, default="davis")
    parser.add_argument("--data_base_path", type=Path, default=None)
    parser.add_argument("--weight_pattern", type=str, default="best_model_fold_{fold}.pt")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--hf_endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--keep_proxy", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--chem_model_name", type=str, default="seyonec/ChemBERTa-zinc-base-v1")
    parser.add_argument("--protein_model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument(
        "--label_priority",
        nargs="+",
        default=["pKd", "pKi", "pIC50"],
        help="Priority used to select one experimental reference value per row.",
    )
    return parser.parse_args()


def configure_huggingface_environment(args):
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    if not args.keep_proxy:
        for key in [
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ]:
            os.environ.pop(key, None)


def clean_protein_sequence(seq):
    return "".join(str(seq).split())


def canonicalize_dataframe(df):
    required = ["SMILES", "Amino_acid_sequence"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"case file is missing required columns: {missing}")

    df = df.copy()
    df["SMILES"] = df["SMILES"].astype(str).str.strip()
    df["Amino_acid_sequence"] = df["Amino_acid_sequence"].map(clean_protein_sequence)
    df = df[(df["SMILES"] != "") & (df["Amino_acid_sequence"] != "")]
    df = df.reset_index(drop=True)
    return df


def select_observed_value(row, priority):
    for col in priority:
        if col not in row.index:
            continue
        value = pd.to_numeric(row[col], errors="coerce")
        if pd.notna(value) and float(value) > 0:
            return float(value), col
    return np.nan, ""


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def try_load_existing_feature_maps(dataset_dir):
    drugs_dir = dataset_dir / "drugs"
    proteins_dir = dataset_dir / "proteins"
    if not drugs_dir.exists() or not proteins_dir.exists():
        return {}, {}

    drug_map = {}
    protein_map = {}
    for index_path in dataset_dir.glob("*_fold_*_index.pt"):
        rows = torch.load(index_path, map_location="cpu")
        for item in rows:
            drug_id = item["drug_id"]
            prot_id = item["prot_id"]
            drug_path = drugs_dir / f"{drug_id}.pt"
            prot_path = proteins_dir / f"{prot_id}.pt"
            if drug_path.exists():
                drug_map.setdefault(drug_id, torch.load(drug_path, map_location="cpu"))
            if prot_path.exists():
                protein_map.setdefault(prot_id, torch.load(prot_path, map_location="cpu"))
    return drug_map, protein_map


def build_case_embeddings(df, cache_dir, code_dir, chem_model_name, protein_model_name):
    sys.path.insert(0, str(code_dir))
    from data_precess import get_chemberta_embeddings, get_protein_embeddings

    cache_dir.mkdir(parents=True, exist_ok=True)
    drug_cache_path = cache_dir / "case_drug_embeddings.pt"
    protein_cache_path = cache_dir / "case_protein_embeddings.pt"

    drug_cache = torch.load(drug_cache_path, map_location="cpu") if drug_cache_path.exists() else {}
    protein_cache = (
        torch.load(protein_cache_path, map_location="cpu") if protein_cache_path.exists() else {}
    )

    smiles_list = df["SMILES"].drop_duplicates().tolist()
    protein_list = df["Amino_acid_sequence"].drop_duplicates().tolist()

    missing_smiles = [smiles for smiles in smiles_list if smiles not in drug_cache]
    missing_proteins = [seq for seq in protein_list if seq not in protein_cache]

    if missing_smiles:
        print(f"Generating ChemBERTa embeddings for {len(missing_smiles)} new drugs...")
        drug_embs = get_chemberta_embeddings(missing_smiles, model_name=chem_model_name)
        for smiles, emb in zip(missing_smiles, drug_embs):
            drug_cache[smiles] = emb.cpu()
        torch.save(drug_cache, drug_cache_path)

    if missing_proteins:
        print(f"Generating ESM2 embeddings for {len(missing_proteins)} new targets...")
        protein_embs = get_protein_embeddings(missing_proteins, model_name=protein_model_name)
        for seq, emb in zip(missing_proteins, protein_embs):
            protein_cache[seq] = emb.cpu()
        torch.save(protein_cache, protein_cache_path)

    return drug_cache, protein_cache


def make_batches(df, drug_cache, protein_cache, batch_size):
    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start : start + batch_size]
        drug_embs = [drug_cache[smiles] for smiles in batch_df["SMILES"]]
        protein_embs = [protein_cache[seq] for seq in batch_df["Amino_acid_sequence"]]

        d_lens = torch.tensor([len(x) for x in drug_embs])
        p_lens = torch.tensor([len(x) for x in protein_embs])
        d_pad = pad_sequence(drug_embs, batch_first=True, padding_value=0.0)
        p_pad = pad_sequence(protein_embs, batch_first=True, padding_value=0.0)

        d_mask = torch.arange(d_pad.shape[1]).expand(len(batch_df), d_pad.shape[1]) >= d_lens.unsqueeze(1)
        p_mask = torch.arange(p_pad.shape[1]).expand(len(batch_df), p_pad.shape[1]) >= p_lens.unsqueeze(1)
        yield start, d_pad, p_pad, d_mask, p_mask


def load_model(code_dir, hidden_dim, weight_path, device):
    sys.path.insert(0, str(code_dir))
    from model import DTA

    model = DTA(hidden_dim=hidden_dim).to(device)
    state = torch.load(weight_path, map_location=device)
    if any(key.startswith("module.") for key in state.keys()):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def predict_one_fold(model, df, drug_cache, protein_cache, batch_size, device, mean_val, std_val):
    preds = np.zeros(len(df), dtype=np.float32)
    with torch.no_grad():
        for start, d_pad, p_pad, d_mask, p_mask in make_batches(df, drug_cache, protein_cache, batch_size):
            d_pad = d_pad.to(device)
            p_pad = p_pad.to(device)
            d_mask = d_mask.to(device)
            p_mask = p_mask.to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                batch_pred = model(d_pad, p_pad, d_mask, p_mask)
            batch_pred = batch_pred.float().cpu().numpy()
            preds[start : start + len(batch_pred)] = batch_pred * std_val + mean_val
    return preds


def main():
    args = parse_args()
    configure_huggingface_environment(args)

    code_dir = args.code_dir.resolve()
    case_file = args.case_file.resolve()
    data_base_path = args.data_base_path or (code_dir / "processed_data")
    dataset_dir = (data_base_path / args.dataset_name).resolve()
    cache_dir = args.cache_dir or (code_dir / "case_analysis_cache")
    output_dir = args.output_dir or (code_dir / "case_analysis_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = canonicalize_dataframe(pd.read_excel(case_file))
    observed = df.apply(lambda row: select_observed_value(row, args.label_priority), axis=1)
    df["experimental_value"] = [x[0] for x in observed]
    df["experimental_endpoint"] = [x[1] for x in observed]

    drug_cache, protein_cache = build_case_embeddings(
        df,
        cache_dir,
        code_dir,
        args.chem_model_name,
        args.protein_model_name,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_predictions = []

    for fold in tqdm(args.folds, desc="Predicting folds"):
        weight_path = code_dir / args.weight_pattern.format(fold=fold)
        if not weight_path.exists():
            raise FileNotFoundError(f"model weight not found: {weight_path}")

        stats_path = dataset_dir / f"fold_{fold}_stats.json"
        if stats_path.exists():
            stats = load_json(stats_path)
            mean_val, std_val = float(stats["mean"]), float(stats["std"])
        else:
            print(f"Warning: {stats_path} not found; using mean=0, std=1.")
            mean_val, std_val = 0.0, 1.0

        model = load_model(code_dir, args.hidden_dim, weight_path, device)
        fold_pred = predict_one_fold(
            model, df, drug_cache, protein_cache, args.batch_size, device, mean_val, std_val
        )
        df[f"pred_fold_{fold}"] = fold_pred
        fold_predictions.append(fold_pred)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pred_matrix = np.vstack(fold_predictions)
    df["pred_mean"] = pred_matrix.mean(axis=0)
    df["pred_std"] = pred_matrix.std(axis=0)
    df["rank_by_pred"] = df["pred_mean"].rank(method="min", ascending=False).astype(int)

    has_observed = df["experimental_value"].notna()
    df["prediction_minus_experiment"] = np.nan
    df.loc[has_observed, "prediction_minus_experiment"] = (
        df.loc[has_observed, "pred_mean"] - df.loc[has_observed, "experimental_value"]
    )

    sort_cols = ["rank_by_pred"]
    if "drug name" in df.columns and "target name" in df.columns:
        display_cols = [
            "rank_by_pred",
            "drug name",
            "drug_CHEMBL",
            "target name",
            "target_CHEMBL",
            "uniprot_ID",
            "target type",
            "MAX PHASE",
            "experimental_endpoint",
            "experimental_value",
            "pred_mean",
            "pred_std",
            "prediction_minus_experiment",
        ]
        display_cols = [col for col in display_cols if col in df.columns]
    else:
        display_cols = list(df.columns)

    result_df = df.sort_values(sort_cols).reset_index(drop=True)
    summary_df = result_df[display_cols]

    xlsx_path = output_dir / "case_analysis_predictions.xlsx"
    csv_path = output_dir / "case_analysis_predictions.csv"
    summary_path = output_dir / "case_analysis_summary.csv"

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="all_predictions", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nTop predicted drug-target pairs:")
    print(summary_df.head(10).to_string(index=False))
    print(f"\nSaved:\n  {xlsx_path}\n  {csv_path}\n  {summary_path}")


if __name__ == "__main__":
    main()

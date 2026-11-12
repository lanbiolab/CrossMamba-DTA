import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import json
import argparse
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, EsmModel


SCRIPT_DIR = Path(__file__).resolve().parent


# =========================
# Argument parser
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="UniProt-annotated target region perturbation analysis."
    )

    parser.add_argument("--code_dir", type=str, default=None)
    parser.add_argument("--case_result_dir", type=str, default=str(SCRIPT_DIR / "case_analysis_results"))
    parser.add_argument("--low_error_file", type=str, default="low_error_cases.csv")
    parser.add_argument("--all_prediction_file", type=str, default="all_case_predictions.csv")
    parser.add_argument("--output_dir", type=str, default=str(SCRIPT_DIR / "uniprot_target_region_perturbation_results"))

    parser.add_argument("--dataset_name", type=str, default="davis")
    parser.add_argument("--data_base_path", type=str, default=str(SCRIPT_DIR / "processed_data"))
    parser.add_argument("--weight_pattern", type=str, default="kiba_best_model_fold_{fold}.pt")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--top_n_cases", type=int, default=3)

    parser.add_argument("--chem_model_name", type=str, default="seyonec/ChemBERTa-zinc-base-v1")
    parser.add_argument("--protein_model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--cache_dir", type=str, default=str(SCRIPT_DIR / "case_analysis_cache"))

    parser.add_argument("--mask_token", type=str, default="X")
    parser.add_argument("--site_flank", type=int, default=5)
    parser.add_argument("--max_region_length", type=int, default=40)
    parser.add_argument("--split_long_regions", action="store_true")

    parser.add_argument("--hf_endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--offline", action="store_true")

    return parser.parse_args()


def configure_environment(args):
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"


def resolve_code_dir(args):
    candidates = []

    if args.code_dir is not None:
        candidates.append(Path(args.code_dir))

    candidates.extend([
        SCRIPT_DIR,
        SCRIPT_DIR.parent / "code",
    ])

    for p in candidates:
        if (p / "model.py").exists():
            sys.path.insert(0, str(p.resolve()))
            return p.resolve()

    raise FileNotFoundError("Cannot find model.py. Please set --code_dir.")


# =========================
# File loading
# =========================
def read_csv_safely(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def load_selected_cases(args):
    case_result_dir = Path(args.case_result_dir)
    low_error_path = case_result_dir / args.low_error_file
    all_prediction_path = case_result_dir / args.all_prediction_file

    if not low_error_path.exists():
        raise FileNotFoundError(f"Cannot find low-error file: {low_error_path}")

    low_df = read_csv_safely(low_error_path)

    if "absolute_error" in low_df.columns:
        low_df = low_df.sort_values("absolute_error", ascending=True)

    selected_df = low_df.head(args.top_n_cases).copy()

    required_cols = ["SMILES", "Amino_acid_sequence", "uniprot_ID"]
    missing = [c for c in required_cols if c not in selected_df.columns]

    if missing:
        if not all_prediction_path.exists():
            raise FileNotFoundError(f"Cannot recover {missing}; missing {all_prediction_path}")

        all_df = read_csv_safely(all_prediction_path)

        recover_cols = [
            "case_index", "SMILES", "Amino_acid_sequence",
            "uniprot_ID", "pKd", "pKi", "pIC50"
        ]
        recover_cols = [c for c in recover_cols if c in all_df.columns]

        if "case_index" in selected_df.columns and "case_index" in all_df.columns:
            selected_df = selected_df.merge(
                all_df[recover_cols],
                on="case_index",
                how="left",
                suffixes=("", "_from_all")
            )
        else:
            key_cols = ["drug_CHEMBL", "target_CHEMBL"]
            selected_df = selected_df.merge(
                all_df[key_cols + [c for c in recover_cols if c not in key_cols]],
                on=key_cols,
                how="left",
                suffixes=("", "_from_all")
            )

    selected_df["SMILES"] = selected_df["SMILES"].astype(str).str.strip()
    selected_df["Amino_acid_sequence"] = (
        selected_df["Amino_acid_sequence"]
        .astype(str)
        .map(lambda x: "".join(x.split()))
    )
    selected_df["uniprot_ID"] = selected_df["uniprot_ID"].astype(str).str.strip()

    return selected_df.reset_index(drop=True)


# =========================
# UniProt annotation
# =========================
KEEP_FEATURE_TYPES = {
    "Transmembrane",
    "Topological domain",
    "Intramembrane",
    "Domain",
    "Region",
    "Compositional bias",
    "Repeat",
    "Motif",
    "Active site",
    "Binding site",
    "Site",
    "Modified residue",
    "Disulfide bond",
    "Glycosylation",
    "Lipidation",
}


def fetch_uniprot_json(uniprot_id, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"uniprot_{uniprot_id}.json"

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to fetch UniProt annotation for {uniprot_id}: {e}")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data


def get_location_value(pos):
    if isinstance(pos, dict):
        return int(pos.get("value"))
    return int(pos)


def normalize_region(start_1based, end_1based, seq_len, site_flank=5):
    if start_1based == end_1based:
        start = max(1, start_1based - site_flank)
        end = min(seq_len, end_1based + site_flank)
    else:
        start = max(1, start_1based)
        end = min(seq_len, end_1based)

    return start, end


def split_region_if_needed(region, max_len):
    start = region["start"]
    end = region["end"]
    length = end - start + 1

    if length <= max_len:
        return [region]

    pieces = []
    current = start

    while current <= end:
        piece_end = min(current + max_len - 1, end)
        r = region.copy()
        r["parent_start"] = start
        r["parent_end"] = end
        r["start"] = current
        r["end"] = piece_end
        r["region_label"] = f"{region['region_label']} | subregion {current}-{piece_end}"
        pieces.append(r)
        current = piece_end + 1

    return pieces


def extract_uniprot_regions(uniprot_id, sequence, cache_dir, site_flank=5, max_region_length=40, split_long_regions=False):
    data = fetch_uniprot_json(uniprot_id, cache_dir)
    seq_len = len(sequence)

    regions = []

    for feat in data.get("features", []):
        feature_type = feat.get("type", "")

        if feature_type not in KEEP_FEATURE_TYPES:
            continue

        location = feat.get("location", {})
        start_obj = location.get("start")
        end_obj = location.get("end")

        if start_obj is None or end_obj is None:
            continue

        try:
            raw_start = get_location_value(start_obj)
            raw_end = get_location_value(end_obj)
        except Exception:
            continue

        start, end = normalize_region(raw_start, raw_end, seq_len, site_flank=site_flank)

        if start > end:
            continue

        description = feat.get("description", "")
        evidence = feat.get("evidences", [])

        evidence_codes = []
        evidence_sources = []

        for ev in evidence:
            if "evidenceCode" in ev:
                evidence_codes.append(str(ev["evidenceCode"]))

            source = ev.get("source", {})
            if isinstance(source, dict):
                source_id = source.get("id", "")
                if source_id:
                    evidence_sources.append(str(source_id))

        region = {
            "uniprot_ID": uniprot_id,
            "annotation_source": "UniProt",
            "feature_type": feature_type,
            "description": description,
            "raw_start": raw_start,
            "raw_end": raw_end,
            "start": start,
            "end": end,
            "region_label": f"{feature_type}: {description} ({start}-{end})",
            "evidence_codes": ";".join(sorted(set(evidence_codes))),
            "evidence_sources": ";".join(sorted(set(evidence_sources))),
        }

        if split_long_regions:
            regions.extend(split_region_if_needed(region, max_region_length))
        else:
            regions.append(region)

    unique = []
    seen = set()

    for r in regions:
        key = (r["feature_type"], r["description"], r["start"], r["end"])

        if key in seen:
            continue

        seen.add(key)
        unique.append(r)

    return unique


# =========================
# Model and encoder
# =========================
def remove_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_model(weight_path, hidden_dim, device):
    from model import DTA

    model = DTA(hidden_dim=hidden_dim).to(device)
    state_dict = torch.load(weight_path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_fold_stats(dataset_dir, fold):
    stats_path = dataset_dir / f"fold_{fold}_stats.json"

    if not stats_path.exists():
        print(f"Warning: {stats_path} not found. Use mean=0, std=1.")
        return 0.0, 1.0

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    return float(stats["mean"]), float(stats["std"])


class SequenceEncoder:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        self.cache_dir = Path(args.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.drug_cache_path = self.cache_dir / "uniprot_region_drug_embeddings.pt"
        self.protein_cache_path = self.cache_dir / "uniprot_region_protein_embeddings.pt"

        self.drug_cache = torch.load(self.drug_cache_path, map_location="cpu") if self.drug_cache_path.exists() else {}
        self.protein_cache = torch.load(self.protein_cache_path, map_location="cpu") if self.protein_cache_path.exists() else {}

        self.chem_tokenizer = None
        self.chem_model = None
        self.protein_tokenizer = None
        self.protein_model = None

    def load_chemberta(self):
        if self.chem_tokenizer is None:
            self.chem_tokenizer = AutoTokenizer.from_pretrained(self.args.chem_model_name)
            self.chem_model = AutoModel.from_pretrained(self.args.chem_model_name).to(self.device)
            self.chem_model.eval()

    def load_esm2(self):
        if self.protein_tokenizer is None:
            self.protein_tokenizer = AutoTokenizer.from_pretrained(self.args.protein_model_name)
            self.protein_model = EsmModel.from_pretrained(self.args.protein_model_name).to(self.device)
            self.protein_model.eval()

    def get_drug_embedding(self, smiles):
        if smiles in self.drug_cache:
            return self.drug_cache[smiles]

        self.load_chemberta()

        with torch.no_grad():
            inputs = self.chem_tokenizer(
                smiles,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.chem_model(**inputs)
            emb = outputs.last_hidden_state.squeeze(0)[1:-1, :].cpu()

        self.drug_cache[smiles] = emb
        torch.save(self.drug_cache, self.drug_cache_path)

        return emb

    def get_protein_embedding(self, sequence):
        if sequence in self.protein_cache:
            return self.protein_cache[sequence]

        self.load_esm2()

        with torch.no_grad():
            inputs = self.protein_tokenizer(
                sequence,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=1024
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.protein_model(**inputs)
            emb = outputs.last_hidden_state.squeeze(0)[1:-1, :].cpu()

        self.protein_cache[sequence] = emb
        torch.save(self.protein_cache, self.protein_cache_path)

        return emb


# =========================
# Perturbation and prediction
# =========================
def mask_protein_region(sequence, start_1based, end_1based, mask_token="X"):
    start = start_1based - 1
    end = end_1based

    seq_list = list(sequence)
    seq_list[start:end] = [mask_token] * (end - start)

    return "".join(seq_list)


def predict_single_pair(model, drug_emb, protein_emb, device, mean_val, std_val):
    drug_emb = drug_emb.unsqueeze(0).to(device)
    protein_emb = protein_emb.unsqueeze(0).to(device)

    drug_mask = torch.zeros((1, drug_emb.shape[1]), dtype=torch.bool, device=device)
    protein_mask = torch.zeros((1, protein_emb.shape[1]), dtype=torch.bool, device=device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            pred_norm = model(drug_emb, protein_emb, drug_mask, protein_mask)

    pred = pred_norm.float().cpu().numpy().reshape(-1)[0]
    return float(pred * std_val + mean_val)


def predict_across_folds(smiles, protein_sequence, models, fold_stats, encoder, device):
    drug_emb = encoder.get_drug_embedding(smiles)
    protein_emb = encoder.get_protein_embedding(protein_sequence)

    fold_preds = []

    for fold, model in models.items():
        mean_val, std_val = fold_stats[fold]
        pred = predict_single_pair(model, drug_emb, protein_emb, device, mean_val, std_val)
        fold_preds.append(pred)

    fold_preds = np.asarray(fold_preds, dtype=np.float32)

    return {
        "fold_predictions": fold_preds,
        "mean": float(fold_preds.mean()),
        "std": float(fold_preds.std()),
    }


# =========================
# Visualization
# =========================
def setup_plot_style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["font.size"] = 10


def safe_name(text):
    text = str(text)
    for ch in ["/", "\\", ":", "*", "?", "\"", "<", ">", "|"]:
        text = text.replace(ch, "_")
    return text[:80]


def plot_uniprot_region_ablation(case_df, case_output_dir, title):
    if case_df.empty:
        return

    plot_df = case_df.sort_values("prediction_drop", ascending=True).tail(15)

    labels = [
        f"{row.feature_type}\n{row.start}-{row.end}"
        for _, row in plot_df.iterrows()
    ]

    fig_height = max(4.0, 0.55 * len(plot_df) + 1.0)
    fig, ax = plt.subplots(figsize=(6.6, fig_height))

    colors = ["#3182BD" if x >= 0 else "#E6550D" for x in plot_df["prediction_drop"]]

    ax.barh(labels, plot_df["prediction_drop"], color=colors, alpha=0.9)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_xlabel(r"Prediction drop after UniProt-region masking ($\Delta \hat{y}$)")
    ax.set_ylabel("UniProt-annotated region")
    ax.set_title(title, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    fig.savefig(case_output_dir / "uniprot_region_ablation.pdf", bbox_inches="tight")
    fig.savefig(case_output_dir / "uniprot_region_ablation.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    args = parse_args()
    configure_environment(args)
    code_dir = resolve_code_dir(args)
    setup_plot_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.data_base_path) / args.dataset_name
    uniprot_cache_dir = Path(args.cache_dir) / "uniprot_annotation_cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Code dir: {code_dir}")
    print(f"Using device: {device}")

    selected_cases = load_selected_cases(args)
    print(f"Loaded {len(selected_cases)} selected cases.")

    encoder = SequenceEncoder(args, device)

    models = {}
    fold_stats = {}

    for fold in args.folds:
        weight_path = code_dir / args.weight_pattern.format(fold=fold)

        if not weight_path.exists():
            raise FileNotFoundError(f"Model weight not found: {weight_path}")

        print(f"Loading fold {fold}: {weight_path}")
        models[fold] = load_model(weight_path, args.hidden_dim, device)
        fold_stats[fold] = load_fold_stats(dataset_dir, fold)

    all_results = []

    for row_idx, row in tqdm(selected_cases.iterrows(), total=len(selected_cases), desc="Analyzing cases"):
        case_index = int(row.get("case_index", row_idx))
        drug_name = row.get("drug name", f"drug_{case_index}")
        target_name = row.get("target name", f"target_{case_index}")
        uniprot_id = str(row.get("uniprot_ID", "")).strip()

        smiles = str(row["SMILES"])
        protein_sequence = str(row["Amino_acid_sequence"])

        if not uniprot_id or uniprot_id.lower() == "nan":
            print(f"Skip case {case_index}: missing UniProt ID.")
            continue

        case_dir_name = f"case_{case_index:03d}_{safe_name(drug_name)}__{safe_name(target_name)}"
        case_output_dir = output_dir / case_dir_name
        case_output_dir.mkdir(parents=True, exist_ok=True)

        original_pred = predict_across_folds(
            smiles, protein_sequence, models, fold_stats, encoder, device
        )

        uniprot_regions = extract_uniprot_regions(
            uniprot_id=uniprot_id,
            sequence=protein_sequence,
            cache_dir=uniprot_cache_dir,
            site_flank=args.site_flank,
            max_region_length=args.max_region_length,
            split_long_regions=args.split_long_regions,
        )

        region_records = []

        for region in tqdm(uniprot_regions, desc=f"Case {case_index} UniProt regions", leave=False):
            masked_sequence = mask_protein_region(
                protein_sequence,
                region["start"],
                region["end"],
                mask_token=args.mask_token
            )

            perturbed_pred = predict_across_folds(
                smiles, masked_sequence, models, fold_stats, encoder, device
            )

            prediction_drop = original_pred["mean"] - perturbed_pred["mean"]

            result = {
                "case_index": case_index,
                "drug_name": drug_name,
                "drug_CHEMBL": row.get("drug_CHEMBL", ""),
                "target_name": target_name,
                "target_CHEMBL": row.get("target_CHEMBL", ""),
                "uniprot_ID": uniprot_id,
                "experimental_value": row.get("experimental_value", np.nan),
                "annotation_source": region["annotation_source"],
                "feature_type": region["feature_type"],
                "description": region["description"],
                "raw_start": region["raw_start"],
                "raw_end": region["raw_end"],
                "start": region["start"],
                "end": region["end"],
                "region_label": region["region_label"],
                "sequence_fragment": protein_sequence[region["start"] - 1:region["end"]],
                "evidence_codes": region["evidence_codes"],
                "evidence_sources": region["evidence_sources"],
                "original_pred_mean": original_pred["mean"],
                "original_pred_std": original_pred["std"],
                "perturbed_pred_mean": perturbed_pred["mean"],
                "perturbed_pred_std": perturbed_pred["std"],
                "prediction_drop": prediction_drop,
                "importance_score": max(prediction_drop, 0.0),
            }

            for fold_idx, pred_value in zip(args.folds, original_pred["fold_predictions"]):
                result[f"original_pred_fold_{fold_idx}"] = float(pred_value)

            for fold_idx, pred_value in zip(args.folds, perturbed_pred["fold_predictions"]):
                result[f"perturbed_pred_fold_{fold_idx}"] = float(pred_value)

            region_records.append(result)
            all_results.append(result)

        case_df = pd.DataFrame(region_records)
        case_df.to_csv(case_output_dir / "uniprot_region_perturbation.csv", index=False, encoding="utf-8-sig")

        with open(case_output_dir / "case_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "case_index": case_index,
                "drug_name": str(drug_name),
                "target_name": str(target_name),
                "uniprot_ID": uniprot_id,
                "experimental_value": None if pd.isna(row.get("experimental_value", np.nan)) else float(row.get("experimental_value")),
                "original_prediction_mean": original_pred["mean"],
                "original_prediction_std": original_pred["std"],
                "num_uniprot_regions": int(len(case_df)),
                "uniprot_rest_url": f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json",
            }, f, indent=2, ensure_ascii=False)

        plot_uniprot_region_ablation(
            case_df,
            case_output_dir,
            f"{drug_name} - {target_name}"
        )

    all_df = pd.DataFrame(all_results)

    all_csv = output_dir / "uniprot_region_perturbation_all_cases.csv"
    summary_csv = output_dir / "uniprot_region_perturbation_summary.csv"
    all_xlsx = output_dir / "uniprot_region_perturbation_all_cases.xlsx"

    all_df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    if not all_df.empty:
        summary_df = (
            all_df.sort_values(["case_index", "prediction_drop"], ascending=[True, False])
            .groupby("case_index")
            .head(10)
            .reset_index(drop=True)
        )

        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

        with pd.ExcelWriter(all_xlsx, engine="openpyxl") as writer:
            all_df.to_excel(writer, sheet_name="all_results", index=False)
            summary_df.to_excel(writer, sheet_name="top_uniprot_regions", index=False)

    print("\nSaved results:")
    print(all_csv)
    print(summary_csv)
    print(all_xlsx)


if __name__ == "__main__":
    main()
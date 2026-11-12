import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Fragments, FunctionalGroups
from transformers import AutoTokenizer, AutoModel, EsmModel


SCRIPT_DIR = Path(__file__).resolve().parent


# =========================
# Argument parser
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="RDKit-based functional group perturbation analysis."
    )

    parser.add_argument("--code_dir", type=str, default=None)
    parser.add_argument("--case_result_dir", type=str, default=str(SCRIPT_DIR / "case_analysis_results"))
    parser.add_argument("--low_error_file", type=str, default="low_error_cases.csv")
    parser.add_argument("--all_prediction_file", type=str, default="all_case_predictions.csv")
    parser.add_argument("--output_dir", type=str, default=str(SCRIPT_DIR / "functional_group_perturbation_results"))

    parser.add_argument("--dataset_name", type=str, default="davis")
    parser.add_argument("--data_base_path", type=str, default=str(SCRIPT_DIR / "processed_data"))
    parser.add_argument("--weight_pattern", type=str, default="kiba_best_model_fold_{fold}.pt")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--top_n_cases", type=int, default=3)

    parser.add_argument("--chem_model_name", type=str, default="seyonec/ChemBERTa-zinc-base-v1")
    parser.add_argument("--protein_model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--cache_dir", type=str, default=str(SCRIPT_DIR / "case_analysis_cache"))

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
# Data loading
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

    required_cols = ["SMILES", "Amino_acid_sequence"]
    missing = [c for c in required_cols if c not in selected_df.columns]

    if missing:
        if not all_prediction_path.exists():
            raise FileNotFoundError(f"Cannot recover {missing}; missing {all_prediction_path}")

        all_df = read_csv_safely(all_prediction_path)

        recover_cols = [
            "case_index", "SMILES", "Amino_acid_sequence",
            "pKd", "pKi", "pIC50"
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

    return selected_df.reset_index(drop=True)


# =========================
# RDKit functional group detection
# =========================
def traverse_fg_hierarchy(nodes):
    result = []

    for node in nodes:
        result.append(node)

        if hasattr(node, "children"):
            result.extend(traverse_fg_hierarchy(node.children))

    return result


def detect_by_rdkit_functional_groups(mol):
    records = []

    try:
        hierarchy = FunctionalGroups.BuildFuncGroupHierarchy()
        nodes = traverse_fg_hierarchy(hierarchy)
    except Exception:
        nodes = []

    seen = set()

    for node in nodes:
        name = getattr(node, "name", "")
        smarts = getattr(node, "smarts", "")
        pattern = getattr(node, "pattern", None)

        if not name or pattern is None:
            continue

        matches = mol.GetSubstructMatches(pattern)

        if not matches:
            continue

        atom_indices = sorted(set(i for m in matches for i in m))
        key = ("FunctionalGroups", name, tuple(atom_indices))

        if key in seen:
            continue

        seen.add(key)

        records.append({
            "functional_group": name,
            "source": "RDKit FunctionalGroups",
            "smarts": smarts,
            "num_matches": len(matches),
            "matched_atom_indices": atom_indices,
        })

    return records


def collect_matches(mol, smarts):
    pattern = Chem.MolFromSmarts(smarts)

    if pattern is None:
        return []

    matches = mol.GetSubstructMatches(pattern)

    if not matches:
        return []

    return sorted(set(i for m in matches for i in m))


def collect_lactone_atoms(mol):
    rings = mol.GetRingInfo().AtomRings()
    atom_indices = set()

    for ring in rings:
        ring_set = set(ring)

        for atom_idx in ring:
            atom = mol.GetAtomWithIdx(atom_idx)

            if atom.GetAtomicNum() != 6:
                continue

            has_carbonyl_o = False
            has_ring_o_neighbor = False

            for bond in atom.GetBonds():
                nbr = bond.GetOtherAtom(atom)

                if bond.GetBondTypeAsDouble() == 2.0 and nbr.GetAtomicNum() == 8:
                    has_carbonyl_o = True
                    atom_indices.add(nbr.GetIdx())

                if nbr.GetAtomicNum() == 8 and nbr.GetIdx() in ring_set:
                    has_ring_o_neighbor = True
                    atom_indices.add(nbr.GetIdx())

            if has_carbonyl_o and has_ring_o_neighbor:
                atom_indices.update(ring_set)

    return sorted(atom_indices)


def collect_imidazole_atoms(mol):
    rings = mol.GetRingInfo().AtomRings()
    atom_indices = set()

    for ring in rings:
        if len(ring) != 5:
            continue

        n_count = sum(1 for idx in ring if mol.GetAtomWithIdx(idx).GetAtomicNum() == 7)
        aromatic_count = sum(1 for idx in ring if mol.GetAtomWithIdx(idx).GetIsAromatic())

        if n_count >= 2 and aromatic_count >= 3:
            atom_indices.update(ring)

    return sorted(atom_indices)


def detect_by_rdkit_fragments(mol):
    """
    Use RDKit built-in fragment descriptors to decide whether a functional group exists.
    Atom indices are then recovered using RDKit substructure or ring queries.
    """
    fragment_specs = [
        {
            "name": "ester",
            "rdkit_func": Fragments.fr_ester,
            "atom_getter": lambda m: collect_matches(m, "[CX3](=O)[OX2][#6]"),
        },
        {
            "name": "lactone",
            "rdkit_func": Fragments.fr_lactone,
            "atom_getter": collect_lactone_atoms,
        },
        {
            "name": "carbonyl",
            "rdkit_func": Fragments.fr_C_O,
            "atom_getter": lambda m: collect_matches(m, "[CX3]=[OX1]"),
        },
        {
            "name": "imidazole",
            "rdkit_func": Fragments.fr_imidazole,
            "atom_getter": collect_imidazole_atoms,
        },
        {
            "name": "amine",
            "rdkit_func": Fragments.fr_NH0,
            "atom_getter": lambda m: collect_matches(m, "[NX3;!$(NC=O);!$(NS(=O)=O)]"),
        },
        {
            "name": "hydroxyl",
            "rdkit_func": Fragments.fr_Al_OH,
            "atom_getter": lambda m: collect_matches(m, "[OX2H]"),
        },
        {
            "name": "sulfonamide",
            "rdkit_func": Fragments.fr_sulfonamd,
            "atom_getter": lambda m: collect_matches(m, "S(=O)(=O)N"),
        },
        {
            "name": "sulfone",
            "rdkit_func": Fragments.fr_sulfone,
            "atom_getter": lambda m: collect_matches(m, "S(=O)(=O)"),
        },
    ]

    records = []

    for spec in fragment_specs:
        try:
            count = int(spec["rdkit_func"](mol))
        except Exception:
            count = 0

        if count <= 0:
            continue

        atoms = spec["atom_getter"](mol)

        if len(atoms) == 0:
            continue

        records.append({
            "functional_group": spec["name"],
            "source": f"RDKit Fragments.{spec['rdkit_func'].__name__}",
            "smarts": "",
            "num_matches": count,
            "matched_atom_indices": atoms,
        })

    return records


def merge_functional_group_records(records):
    merged = []
    seen = set()

    for r in records:
        atom_tuple = tuple(sorted(r["matched_atom_indices"]))
        key = (r["functional_group"].lower(), atom_tuple)

        if key in seen:
            continue

        seen.add(key)
        merged.append(r)

    return merged


def detect_rdkit_functional_groups(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    records = []
    records.extend(detect_by_rdkit_fragments(mol))
    records.extend(detect_by_rdkit_functional_groups(mol))

    records = merge_functional_group_records(records)

    return mol, records


def mask_atoms_with_wildcards(mol, atom_indices):
    rw_mol = Chem.RWMol(mol)

    for idx in atom_indices:
        atom = rw_mol.GetAtomWithIdx(int(idx))
        atom.SetAtomicNum(0)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)

    perturbed_mol = rw_mol.GetMol()

    try:
        return Chem.MolToSmiles(perturbed_mol, isomericSmiles=True)
    except Exception:
        return None


def generate_rdkit_functional_group_perturbations(smiles):
    mol, records = detect_rdkit_functional_groups(smiles)

    perturbations = []
    seen = set()

    for r in records:
        perturbed_smiles = mask_atoms_with_wildcards(mol, r["matched_atom_indices"])

        if perturbed_smiles is None or perturbed_smiles == smiles:
            continue

        key = (r["functional_group"], perturbed_smiles)

        if key in seen:
            continue

        seen.add(key)

        perturbations.append({
            "functional_group": r["functional_group"],
            "source": r["source"],
            "smarts": r["smarts"],
            "num_matches": r["num_matches"],
            "matched_atom_indices": ";".join(map(str, r["matched_atom_indices"])),
            "perturbed_smiles": perturbed_smiles,
        })

    return perturbations


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

        self.drug_cache_path = self.cache_dir / "rdkit_fg_drug_embeddings.pt"
        self.protein_cache_path = self.cache_dir / "rdkit_fg_protein_embeddings.pt"

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


def plot_functional_group_ablation(case_df, case_output_dir, title):
    if case_df.empty:
        return

    plot_df = case_df.sort_values("prediction_drop", ascending=True)
    fig_height = max(3.2, 0.45 * len(plot_df) + 1.2)

    fig, ax = plt.subplots(figsize=(6.6, fig_height))

    colors = ["#3182BD" if x >= 0 else "#E6550D" for x in plot_df["prediction_drop"]]

    labels = [
        f"{row.functional_group}\n({row.source})"
        for _, row in plot_df.iterrows()
    ]

    ax.barh(labels, plot_df["prediction_drop"], color=colors, alpha=0.9)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_xlabel(r"Prediction drop after masking ($\Delta \hat{y}$)")
    ax.set_ylabel("RDKit-detected functional group")
    ax.set_title(title, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    fig.savefig(case_output_dir / "rdkit_functional_group_ablation.pdf", bbox_inches="tight")
    fig.savefig(case_output_dir / "rdkit_functional_group_ablation.png", dpi=600, bbox_inches="tight")
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
        smiles = str(row["SMILES"])
        protein_sequence = str(row["Amino_acid_sequence"])

        case_dir_name = f"case_{case_index:03d}_{safe_name(drug_name)}__{safe_name(target_name)}"
        case_output_dir = output_dir / case_dir_name
        case_output_dir.mkdir(parents=True, exist_ok=True)

        original_pred = predict_across_folds(
            smiles, protein_sequence, models, fold_stats, encoder, device
        )

        perturbations = generate_rdkit_functional_group_perturbations(smiles)

        case_results = []

        for item in perturbations:
            perturbed_pred = predict_across_folds(
                item["perturbed_smiles"],
                protein_sequence,
                models,
                fold_stats,
                encoder,
                device
            )

            prediction_drop = original_pred["mean"] - perturbed_pred["mean"]

            result = {
                "case_index": case_index,
                "drug_name": drug_name,
                "drug_CHEMBL": row.get("drug_CHEMBL", ""),
                "target_name": target_name,
                "target_CHEMBL": row.get("target_CHEMBL", ""),
                "uniprot_ID": row.get("uniprot_ID", ""),
                "experimental_value": row.get("experimental_value", np.nan),
                "original_smiles": smiles,
                "functional_group": item["functional_group"],
                "source": item["source"],
                "smarts": item["smarts"],
                "num_matches": item["num_matches"],
                "matched_atom_indices": item["matched_atom_indices"],
                "perturbed_smiles": item["perturbed_smiles"],
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

            case_results.append(result)
            all_results.append(result)

        case_df = pd.DataFrame(case_results)
        case_df.to_csv(case_output_dir / "rdkit_functional_group_ablation.csv", index=False, encoding="utf-8-sig")

        with open(case_output_dir / "case_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "case_index": case_index,
                "drug_name": str(drug_name),
                "target_name": str(target_name),
                "experimental_value": None if pd.isna(row.get("experimental_value", np.nan)) else float(row.get("experimental_value")),
                "original_prediction_mean": original_pred["mean"],
                "original_prediction_std": original_pred["std"],
                "smiles": smiles,
                "num_rdkit_functional_groups": int(len(case_df)),
            }, f, indent=2, ensure_ascii=False)

        plot_functional_group_ablation(
            case_df,
            case_output_dir,
            f"{drug_name} - {target_name}"
        )

    all_df = pd.DataFrame(all_results)

    all_csv = output_dir / "rdkit_functional_group_ablation_all_cases.csv"
    summary_csv = output_dir / "rdkit_functional_group_ablation_summary.csv"
    all_xlsx = output_dir / "rdkit_functional_group_ablation_all_cases.xlsx"

    all_df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    if not all_df.empty:
        summary_df = (
            all_df.sort_values(["case_index", "prediction_drop"], ascending=[True, False])
            .groupby("case_index")
            .head(5)
            .reset_index(drop=True)
        )

        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

        with pd.ExcelWriter(all_xlsx, engine="openpyxl") as writer:
            all_df.to_excel(writer, sheet_name="all_results", index=False)
            summary_df.to_excel(writer, sheet_name="top_groups", index=False)

    print("\nSaved results:")
    print(all_csv)
    print(summary_csv)
    print(all_xlsx)


if __name__ == "__main__":
    main()
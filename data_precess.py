import torch
# import dgl
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import AutoModel, AutoTokenizer, EsmModel


def smiles_to_dgl_graph(smiles_string: str):
    # 请保留原有的 smiles_to_dgl_graph 实现
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None: return None
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0: return None

    # ... (原有特征提取代码) ...
    # 简化占位，请使用你原来的代码
    g = dgl.graph(([], []), num_nodes=num_atoms)
    # 这里需要确保你原来的代码能运行，核心修改只在下面两个Embedding函数
    return g


def get_chemberta_embeddings(smiles_list: list,
                             model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
                             model: AutoModel = None,
                             tokenizer: AutoTokenizer = None) -> list:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tokenizer is None: tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model is None: model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    encoded_embeddings = []

    with torch.no_grad():
        for smile in smiles_list:
            inputs = tokenizer(smile, return_tensors="pt", padding=False, truncation=True, max_length=128)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            outputs = model(**inputs)
            # 【核心修复】：去掉 batch 维度，并使用 [1:-1] 剔除 [CLS] 和 [SEP]，只保留真实原子特征
            seq_embed = outputs.last_hidden_state.squeeze(0)[1:-1, :].cpu()
            encoded_embeddings.append(seq_embed)

    return encoded_embeddings


def get_protein_embeddings(sequences,
                           model: EsmModel = None,
                           tokenizer: AutoTokenizer = None,
                           model_name: str = 'facebook/esm2_t33_650M_UR50D') -> list:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer is None: tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model is None: model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()
    encoded_embeddings = []

    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding=False, truncation=True, max_length=1024)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            outputs = model(**inputs)
            # 【核心修复】：同理，剔除 ESM2 添加的句首/句尾特殊 Token
            seq_embed = outputs.last_hidden_state.squeeze(0)[1:-1, :].cpu()
            encoded_embeddings.append(seq_embed)

    return encoded_embeddings
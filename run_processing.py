import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, EsmModel

# 导入你原有的特征提取函数
try:
    from data_precess import get_chemberta_embeddings, get_protein_embeddings
except ImportError:
    print("错误：无法从 data_precess.py 导入 get_chemberta_embeddings 或 get_protein_embeddings")
    exit()


def process_data_main():
    BASE_DIR = '.'
    # 对应你 row_data_division.py 生成的路径
    SPLITS_DIR = os.path.join(BASE_DIR, 'dataset', 'splits_dataset')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'processed_data')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    DATASETS = ['kiba']
    N_FOLDS = 5
    BATCH_SIZE = 32  # 预计算时的 batch 大小
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 阶段一: 收集全局唯一的 SMILES 和 Protein ---
    print("\n[阶段一] 正在收集唯一序列...")
    all_dfs = []
    for d_name in DATASETS:
        for fold in range(N_FOLDS):
            fold_dir = os.path.join(SPLITS_DIR, d_name, f'fold_{fold}')
            for s_type in ['train', 'val', 'test']:
                path = os.path.join(fold_dir, f'{s_type}.csv')
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df['D_NAME'] = d_name
                    df['F_ID'] = fold
                    df['S_TYPE'] = s_type
                    all_dfs.append(df)

    if not all_dfs:
        print("错误：未找到 CSV 文件，请检查路径。")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    unique_smiles = full_df['COMPOUND_SMILES'].unique().tolist()
    unique_proteins = full_df['PROTEIN_SEQUENCE'].unique().tolist()
    print(f"统计：{len(unique_smiles)} 种药物, {len(unique_proteins)} 种蛋白质。")

    # --- 阶段二: 预计算 Embeddings ---
    print(f"\n[阶段二] 正在预计算 Embeddings (使用设备: {device})...")

    # 1. 药物侧: ChemBERTa
    chem_model_name = "seyonec/ChemBERTa-zinc-base-v1"
    chem_tokenizer = AutoTokenizer.from_pretrained(chem_model_name)
    chem_model = AutoModel.from_pretrained(chem_model_name).to(device)

    smiles_to_embedding = {}
    for i in tqdm(range(0, len(unique_smiles), BATCH_SIZE), desc="计算 ChemBERTa 嵌入"):
        batch = unique_smiles[i:i + BATCH_SIZE]
        # 调用你 data_precess.py 中的函数
        embs = get_chemberta_embeddings(batch, model=chem_model, tokenizer=chem_tokenizer)
        for s, e in zip(batch, embs):
            smiles_to_embedding[s] = e.cpu()

    del chem_model  # 释放显存给 ESM2

    # 2. 蛋白侧: ESM2
    protein_model_name = 'facebook/esm2_t33_650M_UR50D'
    protein_tokenizer = AutoTokenizer.from_pretrained(protein_model_name)
    protein_model = EsmModel.from_pretrained(protein_model_name).to(device)

    protein_to_embedding = {}
    for i in tqdm(range(0, len(unique_proteins), BATCH_SIZE), desc="计算 ESM2 嵌入"):
        batch = unique_proteins[i:i + BATCH_SIZE]
        embs = get_protein_embeddings(batch, model=protein_model, tokenizer=protein_tokenizer)
        for p, e in zip(batch, embs):
            protein_to_embedding[p] = e.cpu()

    del protein_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- 阶段三: 保存特征池与索引 ---
    print("\n[阶段三] 正在组装索引与特征池...")
    for d_name in DATASETS:
        d_out_dir = os.path.join(OUTPUT_DIR, d_name)
        os.makedirs(os.path.join(d_out_dir, 'drugs'), exist_ok=True)
        os.makedirs(os.path.join(d_out_dir, 'proteins'), exist_ok=True)

        # 提取该数据集相关的子集
        d_df = full_df[full_df['D_NAME'] == d_name]
        d_smiles = d_df['COMPOUND_SMILES'].unique().tolist()
        d_prots = d_df['PROTEIN_SEQUENCE'].unique().tolist()

        # 映射并保存药物
        s2id = {s: i for i, s in enumerate(d_smiles)}
        for s, idx in s2id.items():
            torch.save(smiles_to_embedding[s], os.path.join(d_out_dir, 'drugs', f'{idx}.pt'))

        # 映射并保存蛋白
        p2id = {p: i for i, p in enumerate(d_prots)}
        for p, idx in p2id.items():
            torch.save(protein_to_embedding[p], os.path.join(d_out_dir, 'proteins', f'{idx}.pt'))

        # 保存元数据
        meta = {
            "drug_dim": smiles_to_embedding[d_smiles[0]].shape[-1],
            "prot_dim": protein_to_embedding[d_prots[0]].shape[-1],
            "drug_encoder": "ChemBERTa",
            "prot_encoder": "ESM2"
        }
        with open(os.path.join(d_out_dir, 'feature_meta.json'), 'w') as f:
            json.dump(meta, f, indent=4)

        # 按 Fold 生成索引
        for fold in range(N_FOLDS):
            f_df = d_df[d_df['F_ID'] == fold]
            train_f = f_df[f_df['S_TYPE'] == 'train']

            if train_f.empty: continue

            # 计算归一化参数
            mean, std = float(train_f['REG_LABEL'].mean()), float(train_f['REG_LABEL'].std())
            with open(os.path.join(d_out_dir, f'fold_{fold}_stats.json'), 'w') as f:
                json.dump({'mean': mean, 'std': std}, f, indent=4)

            # 生成索引文件
            for s_type in ['train', 'val', 'test']:
                split_df = f_df[f_df['S_TYPE'] == s_type]
                index_data = []
                for row in split_df.itertuples():
                    index_data.append({
                        'drug_id': s2id[row.COMPOUND_SMILES],
                        'prot_id': p2id[row.PROTEIN_SEQUENCE],
                        'label': torch.tensor((row.REG_LABEL - mean) / std, dtype=torch.float32)
                    })
                torch.save(index_data, os.path.join(d_out_dir, f'{s_type}_fold_{fold}_index.pt'))
                print(f"完成: {d_name} Fold {fold} {s_type} (条目: {len(index_data)})")


if __name__ == '__main__':
    process_data_main()
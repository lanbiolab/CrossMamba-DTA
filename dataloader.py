import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import os
from tqdm import tqdm


class DTADataset(Dataset):
    def __init__(self, index_file_path: str, drugs_dir: str, proteins_dir: str):
        super(DTADataset, self).__init__()
        self.data = torch.load(index_file_path, map_location='cpu')

        drug_ids = set(item['drug_id'] for item in self.data)
        prot_ids = set(item['prot_id'] for item in self.data)

        self.drug_cache = {}
        self.prot_cache = {}

        print(f"\n正在将特征预加载到内存中 (共有 {len(drug_ids)} 种药物, {len(prot_ids)} 种蛋白)...")
        for d_id in tqdm(drug_ids, desc="预加载药物特征", leave=False):
            drug_path = os.path.join(drugs_dir, f"{d_id}.pt")
            self.drug_cache[d_id] = torch.load(drug_path, map_location='cpu')

        for p_id in tqdm(prot_ids, desc="预加载蛋白特征", leave=False):
            prot_path = os.path.join(proteins_dir, f"{p_id}.pt")
            self.prot_cache[p_id] = torch.load(prot_path, map_location='cpu')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'drug_embedding': self.drug_cache[item['drug_id']],
            'protein_embedding': self.prot_cache[item['prot_id']],
            'label': item['label']
        }


def collate_fn(samples):
    d_embs = [s['drug_embedding'] for s in samples]
    p_embs = [s['protein_embedding'] for s in samples]
    batched_labels = torch.stack([s['label'] for s in samples])

    d_lens = torch.tensor([len(d) for d in d_embs])
    p_lens = torch.tensor([len(p) for p in p_embs])

    drug_padded = pad_sequence(d_embs, batch_first=True, padding_value=0.0)
    protein_padded = pad_sequence(p_embs, batch_first=True, padding_value=0.0)

    batch_size, max_d_len = drug_padded.shape[0], drug_padded.shape[1]
    batch_size, max_p_len = protein_padded.shape[0], protein_padded.shape[1]

    d_mask = torch.arange(max_d_len).expand(batch_size, max_d_len) >= d_lens.unsqueeze(1)
    p_mask = torch.arange(max_p_len).expand(batch_size, max_p_len) >= p_lens.unsqueeze(1)

    return drug_padded, protein_padded, d_mask, p_mask, batched_labels
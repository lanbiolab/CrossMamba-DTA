import torch
import torch.nn.functional as F
from torch import nn
from model_utils import MotifConvAdapter, GatedCrossMambaInteraction


# 【方案 1 核心】：GeM 池化层，无缝替代 MeanPooling
class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeMPooling, self).__init__()
        # p 是一个可学习的参数，初始化为 3。自动寻找平均与最大的平衡点
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x, mask):
        # mask 中 True 表示 padding 占位符
        valid_mask = (~mask).float().unsqueeze(-1)  # [batch, seq_len, 1]
        x = x * valid_mask

        # Clamp 防止底数为负或 0 导致反向传播 NaN (相当于应用了一个微小的 ReLU)
        # 然后求 p 次方
        x = x.clamp(min=self.eps).pow(self.p)

        # 沿序列维度求和
        sum_x = x.sum(dim=1)
        # 计算有效序列长度
        valid_count = valid_mask.sum(dim=1).clamp(min=1e-9)

        # 求均值后开 p 次方根
        return (sum_x / valid_count).pow(1.0 / self.p)


class DTA(nn.Module):
    def __init__(self, protein_seq_feat=1280, drug_seq_feat=768, hidden_dim=512):
        super(DTA, self).__init__()

        # 1. 局部特征提取层
        self.prot_adapter = MotifConvAdapter(protein_seq_feat, hidden_dim)
        self.drug_adapter = MotifConvAdapter(drug_seq_feat, hidden_dim)

        # 2. Mamba 全局跨模态交互层
        self.interaction = GatedCrossMambaInteraction(hidden_dim=hidden_dim, dropout=0.1)

        # 3. 【方案 1】：换用 GeM 池化，精准聚焦活性位点，避免特征被稀释
        self.drug_pool = GeMPooling()
        self.prot_pool = GeMPooling()

        # 4. 亲和力预测器
        # 【方案 2】：因为拼接了 [d, p, d*p, |d-p|]，维度变成了 4 倍
        combined_dim = hidden_dim * 2

        self.affinity_predictor = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 1)  # 纯粹的 1D 输出
        )

    def forward(self, drug_seq_feats, protein_seq_feats, d_mask, p_mask):
        p_feat = self.prot_adapter(protein_seq_feats)
        d_feat = self.drug_adapter(drug_seq_feats)

        # 训练阶段注入极小的高斯噪声防止死记硬背
        if self.training:
            noise_std = 0.01
            p_feat = p_feat + torch.randn_like(p_feat) * noise_std
            d_feat = d_feat + torch.randn_like(d_feat) * noise_std

        # Mamba 双向相互作用模拟
        d_interact, p_interact = self.interaction(d_feat, p_feat, d_mask, p_mask)

        # 特征压缩 (使用 GeM 提取尖锐特征)
        d_pooled = self.drug_pool(d_interact, d_mask)
        p_pooled = self.prot_pool(p_interact, p_mask)

        # 【方案 2 核心】：启发式特征融合 (SBERT 机制)
        d_mul_p = d_pooled * p_pooled  # 逐元素乘积 (模拟结合部位的空间重叠)
        d_sub_p = torch.abs(d_pooled - p_pooled)  # 绝对差异 (模拟两种模态特征的距离)

        # 将原始特征、乘积、差值一起拼接喂给 MLP
        combined_features = torch.cat([d_pooled, p_pooled], dim=-1)
        output = self.affinity_predictor(combined_features)

        return output.squeeze(-1)
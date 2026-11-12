import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("请先安装 mamba 环境: pip install causal-conv1d mamba-ssm")


class MotifConvAdapter(nn.Module):
    """
    使用 1D-CNN 提取局部物理化学 Motifs，并统一特征维度。
    卷积操作本身对顺序敏感，起到了隐式位置编码的作用。
    """

    def __init__(self, in_features, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=hidden_dim,
            kernel_size=5,
            padding=2
        )

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        residual = x
        x = self.feed_forward(x)
        return self.norm(x + residual)


class BiMambaBlock(nn.Module):
    """
    双向 Mamba 块：生物序列没有严格时间因果关系，必须正反向同时扫描
    """

    def __init__(self, d_model, d_state=8, d_conv=4, expand=2):
        super().__init__()
        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        # 正向扫描
        out_fwd = self.forward_mamba(x)
        # 反向扫描 (序列翻转 -> 扫描 -> 翻转回来)
        out_bwd = self.backward_mamba(torch.flip(x, dims=[1]))
        out_bwd = torch.flip(out_bwd, dims=[1])

        # 融合双向信息
        return out_fwd + out_bwd


class GatedCrossMambaInteraction(nn.Module):
    """
    【创新架构】：基于双向状态空间模型 (Bi-SSM) 的跨模态交互层
    替代了传统的 O(N^2) Cross-Attention，实现线性时间复杂度的全局对齐。
    """

    def __init__(self, hidden_dim=512, dropout=0.1):
        super().__init__()
        # 使用双向 Mamba 作为序列混合器 (Mixer)
        self.mamba_mixer = BiMambaBlock(d_model=hidden_dim)

        # 依然保留你优秀的物理直觉：门控机制模拟“诱导契合”
        self.gate_d = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.gate_p = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

        self.norm_d = nn.LayerNorm(hidden_dim)
        self.norm_p = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, d_seq, p_seq, d_mask, p_mask):
        # 1. Padding 掩码处理 (Mamba 没有类似 Attention 的 mask 参数，需将 padding 物理置 0)
        d_seq = d_seq.masked_fill(d_mask.unsqueeze(-1), 0.0)
        p_seq = p_seq.masked_fill(p_mask.unsqueeze(-1), 0.0)

        # 2. 序列拼接：让 SSM 的隐状态 (Hidden State) 在药物和靶点之间流淌
        # 拼接后维度: [batch_size, d_len + p_len, hidden_dim]
        d_len = d_seq.size(1)
        combined_seq = torch.cat([d_seq, p_seq], dim=1)

        # 3. Mamba 核心交互
        mixed_seq = self.mamba_mixer(combined_seq)
        mixed_seq = self.dropout(mixed_seq)

        # 4. 特征拆分
        d_mixed = mixed_seq[:, :d_len, :]
        p_mixed = mixed_seq[:, d_len:, :]

        # 5. 门控融合
        g_d = self.gate_d(torch.cat([d_seq, d_mixed], dim=-1))
        g_p = self.gate_p(torch.cat([p_seq, p_mixed], dim=-1))

        d_out = self.norm_d(d_seq + g_d * d_mixed)
        p_out = self.norm_p(p_seq + g_p * p_mixed)

        # 6. 再次清理 Padding 产生的噪声
        d_out = d_out.masked_fill(d_mask.unsqueeze(-1), 0.0)
        p_out = p_out.masked_fill(p_mask.unsqueeze(-1), 0.0)

        return d_out, p_out


class AttentionPooling(nn.Module):
    """动态注意力池化：过滤掉冗余背景，聚焦于实际发生结合的位点"""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x, mask):
        attn_weights = self.attention(x).squeeze(-1)
        # 用极小值填充 padding，确保 softmax 后该位置权重为 0
        attn_weights = attn_weights.masked_fill(mask, -1e4)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # 加权求和: [batch, seq_len, dim] -> [batch, dim]
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        return pooled


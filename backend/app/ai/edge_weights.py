import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

class EdgeWeightsLayerV3(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout_rate: float = 0.2,
        temperature: float = 2.0
    ) -> None:

        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.temperature = temperature

        total_hidden_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LeakyReLU(0.2)
        )

        self.qk_proj = nn.Linear(in_dim // 2, total_hidden_dim * 2)

        self.dropout = nn.Dropout(dropout_rate)
        self.tanh = nn.Tanh()

    def forward(self, feats: Tensor) -> Tensor:
        N = feats.size(0)

        # (N, in_dim) -> (N, in_dim)
        h = self.backbone(feats)

        # (N, 2 * H*D) -> tách (Q, K)
        qk = self.qk_proj(h)  # (N, 2 * total_hidden_dim)
        Q, K = qk.chunk(2, dim=-1)

        # [N, H, D]
        Q = Q.view(N, self.num_heads, self.head_dim)
        K = K.view(N, self.num_heads, self.head_dim)

        # Q = F.normalize(Q, dim=-1)   
        # K = F.normalize(K, dim=-1)

        Q = self.dropout(Q)
        K = self.dropout(K)

        # Dot-product similarity
        # (i,h,d) * (j,h,d) -> (i,j,h)
        attn_scores = torch.einsum('ihd,jhd->ijh', Q, K) * self.scale

        # Mean heads 
        H_mean = attn_scores.mean(dim=-1)  # (N, N, H) -> (N, N)
        # H_mean = self.dropout(H_mean)
        return self.tanh(H_mean / self.temperature)
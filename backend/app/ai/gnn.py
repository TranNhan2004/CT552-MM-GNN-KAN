import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


#=======================================================================================
class SignedGATLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, heads=1, concat=True, dropout_rate=0.0, negative_slope=0.2):
        super().__init__(aggr='add', node_dim=0)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout_rate
        self.negative_slope = negative_slope
        
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def message(
        self, 
        x_i: Tensor, 
        x_j: Tensor, 
        edge_attr: Tensor, 
        index: Tensor, 
        ptr: Tensor, 
        size_i: int
    ) -> Tensor:
        
        # 1. Tính điểm attention thô 'e'
        # e shape: [E, H]
        e = (x_j * self.att_src).sum(-1) + (x_i * self.att_dst).sum(-1)
        e = F.leaky_relu(e, self.negative_slope)
        
        # 2. Tách độ lớn (mag) và dấu (sign) từ trọng số CW
        sign = edge_attr.sign().unsqueeze(-1)  # Shape: [E, 1]
        mag = edge_attr.abs().unsqueeze(-1)    # Shape: [E, 1]

        # 3. Tính "điểm số có trọng số" (signed score)
        # (Đây là điểm số cuối cùng, có thể âm)
        # e_signed shape: [E, H] * [E, 1] -> [E, H]
        e_signed = e * mag * sign 
        
        # 4. Lấy "độ lớn" (magnitude) của điểm số
        # e_magnitude shape: [E, H]
        e_magnitude = e_signed.abs()

        alpha_magnitude = softmax(e_magnitude, index, ptr, size_i) # Shape: [E, H]
        
        # 5. Áp dụng lại "Dấu"
        # (Lấy dấu của e_signed, không phải của edge_attr thô)
        alpha_sign = e_signed.sign()
        alpha_final = alpha_magnitude * alpha_sign
        
        alpha_final = F.dropout(alpha_final, p=self.dropout, training=self.training)
        
        # 6. Tạo thông điệp (message)
        # x_j [E, H, C] * alpha_final.unsqueeze(-1) [E, H, 1] -> [E, H, C]
        return x_j * alpha_final.unsqueeze(-1)
    
    def forward(self, X: Tensor, edge_index: Tensor, edge_weights: Tensor = None) -> Tensor:
        N, H, C = X.size(0), self.heads, self.out_dim
        h = self.lin(X).view(N, H, C)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.propagate(edge_index, x=h, edge_attr=edge_weights)
        if self.concat:
            out = out.view(N, H * C)
        else:
            out = out.mean(dim=1)
        return out


#=======================================================================================
class MultiGATLayerV3(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc_residual = nn.Linear(in_dim, out_dim)

        self.gat1 = SignedGATLayer(in_dim=in_dim, out_dim=256, heads=4, concat=True, dropout_rate=dropout_rate)
        self.gat2 = SignedGATLayer(in_dim=256*4, out_dim=out_dim, heads=2, concat=False, dropout_rate=dropout_rate)

        self.norm_input = nn.LayerNorm(in_dim)
        self.norm_hidden = nn.LayerNorm(out_dim)
        self.fc_combine = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.elu = nn.ELU()

    def forward(self, X: Tensor, CW: Tensor) -> Tensor:
        edge_index_all = CW.nonzero().t().contiguous()
        edge_weights_all = CW[edge_index_all[0], edge_index_all[1]]

        X_normed = self.norm_input(X)
        H = self.gat1(X_normed, edge_index_all, edge_weights_all)
        H = self.elu(H)
        H = self.gat2(H, edge_index_all, edge_weights_all)
        H = self.elu(H)

        H = self.norm_hidden(H)
        H = self.fc_combine(H)
        H = self.elu(H)

        X_residual = self.fc_residual(X)
        return X_residual + self.dropout(H)

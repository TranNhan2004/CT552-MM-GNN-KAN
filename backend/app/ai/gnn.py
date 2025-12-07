import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from typing import Tuple


#=======================================================================================
class GATLayer(MessagePassing):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        heads: int = 1, 
        concat: bool = True, 
        dropout_rate: float = 0.0, 
        negative_slope: float = 0.2
    ) -> None:
        
        super().__init__(aggr='add', node_dim=0)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout_rate
        
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
    
        e = (x_j * self.att_src).sum(-1) + (x_i * self.att_dst).sum(-1)
        alpha = F.leaky_relu(e, self.negative_slope)
        
        if edge_attr is not None:
            alpha = alpha * edge_attr.view(-1, 1)
        
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j * alpha.unsqueeze(-1)
    
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

        self.gat1 = GATLayer(in_dim=in_dim, out_dim=256, heads=4, concat=True, dropout_rate=dropout_rate)
        self.gat2 = GATLayer(in_dim=256*4, out_dim=out_dim, heads=2, concat=False, dropout_rate=dropout_rate)

        self.norm_input = nn.LayerNorm(in_dim)
        self.norm_hidden = nn.LayerNorm(out_dim)
        self.fc_combine = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.elu = nn.ELU()

    def forward(self, X: Tensor, CW: Tensor) -> Tensor:
        pos_w = torch.clamp(CW, min=0.0)
        edge_index_all = pos_w.nonzero().t().contiguous()
        edge_weights_all = pos_w[edge_index_all[0], edge_index_all[1]]

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

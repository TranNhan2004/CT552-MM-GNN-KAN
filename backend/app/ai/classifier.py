import torch
import torch.nn as nn

from torch import Tensor
from fastkan.fastkan import FastKAN

#=======================================================================================
class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, negative_slope: float = 0.15):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LeakyReLU(negative_slope),
            nn.Linear(in_dim // 2, num_classes),
        )

    def forward(self, X: Tensor, graph_weights: Tensor) -> Tensor:
        node_logits = self.mlp(X)
        pos_w = torch.clamp(graph_weights, min=0.0)
        node_w = pos_w.sum(dim=1)
        w = node_w.unsqueeze(-1)  
        graph_logits = (node_logits * w).sum(0, keepdim=True) / (w.sum() + 1e-8)
        return graph_logits


#=======================================================================================
class FastKANClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fk = FastKAN(
            layers_hidden=[in_dim, in_dim // 2, num_classes],
            num_grids=6
        )

    def forward(self, X: Tensor, graph_weights: Tensor) -> Tensor:
        node_logits = self.fk(X)                                            # [N, num_classes]
        pos_w = torch.clamp(graph_weights, min=0.0)
        node_w = pos_w.sum(dim=1)                                        # [N]
        w = node_w.unsqueeze(-1)                                         # [N, 1]
        graph_logits = (node_logits * w).sum(0, keepdim=True) / (w.sum() + 1e-8)
        return graph_logits


#=======================================================================================
class MeanLogitsClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    
    def forward(self, X: Tensor) -> Tensor:
        logits_per_image = self.fc(X)  # [Num_Images, Num_Classes]                    
        graph_logits = logits_per_image.mean(dim=0, keepdim=True) # [1, Num_Classes]
        return graph_logits
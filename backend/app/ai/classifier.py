import torch.nn as nn

from fastkan.fastkan import FastKAN


#=======================================================================================
class MLPClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, negative_slope=0.15):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LeakyReLU(negative_slope),
            nn.Linear(in_dim // 2, num_classes),
        )


    def forward(self, X, graph_weights):
        node_logits = self.mlp(X)
        node_w = graph_weights.sum(dim=1) 
        w = node_w.unsqueeze(-1)  
        graph_logits = (node_logits * w).sum(0, keepdim=True) / (w.sum() + 1e-8)
        return graph_logits


#=======================================================================================
class FastKANClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fk = FastKAN(
            layers_hidden=[in_dim, in_dim // 2, num_classes],
            num_grids=6
        )

    def forward(self, X, graph_weights):
        node_logits = self.fk(X)            # [N, num_classes]
        node_w = graph_weights.sum(dim=1)   # [N]
        w = node_w.unsqueeze(-1)            # [N, 1]
        graph_logits = (node_logits * w).sum(0, keepdim=True) / (w.sum() + 1e-8)
        return graph_logits

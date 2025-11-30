import torch
import torch.nn as nn

from torch import Tensor

from ..ai.image_helpers import ImageHelpers
from ..models.image import ImageModelType

#=======================================================================================
class ExtractImageFeaturesLayer(nn.Module):
    def __init__(self, image_model_name: ImageModelType, in_channels: int, out_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.backbone_model = ImageHelpers.get_backbone_model(image_model_name)
        self.fc = nn.Linear(self.backbone_model.out_features, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, X: Tensor) -> Tensor:
        H = self.backbone(X)
        H = self.fc(H)
        return self.dropout(H)


#=======================================================================================
class ExtractTextFeaturesLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float = 0.2, negative_slope: float = 0.2):
        super().__init__()
        self.fc_residual_word = nn.Linear(in_dim, out_dim)
        self.fc_combine = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(dropout_rate),
        )
        
    def forward(self, X_word: Tensor, X_sent: Tensor) -> Tensor:
        H = self.fc_combine(torch.cat([X_word, X_sent], dim=1))
        X_word_residual = self.fc_residual_word(X_word)
        return H + X_word_residual


#=======================================================================================
class ExtractAudioFeaturesLayer(nn.Module):
    def __init__(self, mfcc_dim: int, out_dim: int, dropout_rate: float = 0.2):
        super().__init__()        
        self.fc_residual_mfcc = nn.Linear(mfcc_dim, out_dim)
        self.fc_combine = nn.Sequential(
            nn.Linear(mfcc_dim * 3, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        

    def forward(self, X):
        # X shape: [N, C=3, F, T]
        N, C, F, T = X.shape

        # [N, C, F, T] -> [N, T, C, F] -> [N, T, C*F]
        H = X.permute(0, 3, 1, 2).reshape(N, T, -1)
        
        # H shape: [N, T, 39] -> [N, T, O]
        H = self.fc_combine(H)

        # [N, T, O] -> [N, O]
        H_pooled = H.mean(dim=1) 

        X_mfcc = X[:, 0, :, :].permute(0, 2, 1) # -> [N, T, F]
        X_mfcc = self.fc_residual_mfcc(X_mfcc)
        X_mfcc_pooled = X_mfcc.mean(dim=1)

        return H_pooled + X_mfcc_pooled


#=======================================================================================
class ShareProjectionLayerV2(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float = 0.2, negative_slope: float = 0.15) -> None:
        super().__init__()

        self.fc1 = nn.Linear(in_dim, in_dim + 64)
        self.fc2 = nn.Linear(in_dim + 64, out_dim)
        
        self.norm_input = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.modality_codes = nn.ParameterDict({
            "images": nn.Parameter(torch.randn(out_dim)),
            "texts": nn.Parameter(torch.randn(out_dim)),
            "audios": nn.Parameter(torch.randn(out_dim)),
        })
        self.fc_residual = nn.Linear(in_dim, out_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, X: Tensor, modal: str) -> Tensor:
        X_normed = self.norm_input(X)
        H = self.fc1(X_normed)
        H = self.leaky_relu(H)

        H = self.fc2(H)
        H = self.leaky_relu(H)
        
        M = self.modality_codes[modal].expand_as(H)
        
        H_out = self.leaky_relu(H + M)
        H_out = self.dropout(H_out)
        
        X_residual = self.fc_residual(X) 
        
        return X_residual + H_out

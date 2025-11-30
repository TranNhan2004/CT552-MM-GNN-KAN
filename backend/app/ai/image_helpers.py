import torch.nn as nn
import torchvision.models as tvmodels

from ..models.image import ImageModelType
from torchvision.transforms import Compose
from torchvision.models import (
    ResNet18_Weights,
    RegNet_X_400MF_Weights,
    DenseNet121_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    ShuffleNet_V2_X1_0_Weights
)

class ImageHelpers:
    @staticmethod
    def get_model_transforms(model_name: ImageModelType) -> Compose:
        if model_name == "resnet":
            weights = ResNet18_Weights.DEFAULT
        elif model_name == "regnet":
            weights = RegNet_X_400MF_Weights.DEFAULT
        elif model_name == "densenet":
            weights = DenseNet121_Weights.DEFAULT
        elif model_name == "mobilenetv3large":
            weights = MobileNet_V3_Large_Weights.DEFAULT
        elif model_name == "mobilenetv3small":
            weights = MobileNet_V3_Small_Weights.DEFAULT
        elif model_name == "shufflenet":
            weights = ShuffleNet_V2_X1_0_Weights.DEFAULT
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return weights.transforms()
    
    @staticmethod
    def get_backbone_model(model_name: ImageModelType) -> nn.Module:
        if model_name == "resnet":
            return ResNetBackbone()
        elif model_name == "regnet":
            return RegNetBackbone()
        elif model_name == "densenet":
            return DenseNetBackbone()
        elif model_name == "mobilenetv3large":
            return MobileNetBackbone()
        elif model_name == "mobilenetv3small":
            return MobileNetSmallBackbone()
        elif model_name == "shufflenet":
            return ShuffleNetBackbone()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    


#=======================================================================================
class ResNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = tvmodels.resnet18(weights=weights)
        self.out_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False

        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval() 
                module.train = lambda mode=True: None
        
    def forward(self, X):
        return self.backbone(X)


#=======================================================================================
class RegNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = RegNet_X_400MF_Weights.DEFAULT if pretrained else None
        self.backbone = tvmodels.regnet_x_400mf(weights=weights)
        
        self.out_features = self.backbone.fc.in_features if hasattr(self.backbone, "fc") else 2048
        self.backbone.fc = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False

        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval() 
                module.train = lambda mode=True: None
        
    def forward(self, X):
        return self.backbone(X)


#=======================================================================================
class MobileNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.backbone = tvmodels.mobilenet_v3_large(weights=weights)
        
        self.out_features = self.backbone.features[-1].out_channels
        self.backbone.classifier = nn.Identity()
        
        for param in self.backbone.parameters():
            param.requires_grad = False

        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval() 
                module.train = lambda mode=True: None
    
    def forward(self, X):
        return self.backbone(X)


#=======================================================================================
class MobileNetSmallBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = tvmodels.mobilenet_v3_small(weights=weights)
        
        self.out_features = self.backbone.features[-1].out_channels
        self.backbone.classifier = nn.Identity()
        
        for param in self.backbone.parameters():
            param.requires_grad = False

        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval() 
                module.train = lambda mode=True: None
    
    def forward(self, X):
        return self.backbone(X)


#=======================================================================================
class DenseNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = tvmodels.densenet121(weights=weights)
        self.out_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False

        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                module.train = lambda mode=True: None

    def forward(self, X):
        return self.backbone(X)


#=======================================================================================
class ShuffleNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        self.backbone = tvmodels.shufflenet_v2_x1_0(weights=weights)
        self.out_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False

        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                module.train = lambda mode=True: None

    def forward(self, X):
        return self.backbone(X)

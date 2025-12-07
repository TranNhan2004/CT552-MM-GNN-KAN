from typing import Tuple
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
    def _freeze_backbone(model: nn.Module):
        for p in model.parameters():
            p.requires_grad = False

        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
                m.train = lambda mode=True: None

    @staticmethod
    def get_backbone(model_type: ImageModelType, pretrained: bool = True) -> Tuple[nn.Module, int]:
        if model_type == "resnet":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = tvmodels.resnet18(weights=weights)
            out_features = model.fc.in_features
            model.fc = nn.Identity()
            ImageHelpers._freeze_backbone(model)
            return model, out_features

        if model_type == "regnet":
            weights = RegNet_X_400MF_Weights.DEFAULT if pretrained else None
            model = tvmodels.regnet_x_400mf(weights=weights)
            out_features = model.fc.in_features
            model.fc = nn.Identity()
            ImageHelpers._freeze_backbone(model)
            return model, out_features

        if model_type == "mobilenetv3large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            model = tvmodels.mobilenet_v3_large(weights=weights)
            out_features = model.features[-1].out_channels
            model.classifier = nn.Identity()
            ImageHelpers._freeze_backbone(model)
            return model, out_features

        if model_type == "mobilenetv3small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = tvmodels.mobilenet_v3_small(weights=weights)
            out_features = model.features[-1].out_channels
            model.classifier = nn.Identity()
            ImageHelpers._freeze_backbone(model)
            return model, out_features

        if model_type == "densenet":
            weights = DenseNet121_Weights.DEFAULT if pretrained else None
            model = tvmodels.densenet121(weights=weights)
            out_features = model.classifier.in_features
            model.classifier = nn.Identity()
            ImageHelpers._freeze_backbone(model)
            return model, out_features

        if model_type == "shufflenet":
            weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
            model = tvmodels.shufflenet_v2_x1_0(weights=weights)
            out_features = model.fc.in_features
            model.fc = nn.Identity()
            ImageHelpers._freeze_backbone(model)
            return model, out_features

        raise ValueError(f"Unsupported model type: {model_type}")

    




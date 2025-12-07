import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from typing import Dict, Any

from backend.app.ai.image_helpers import ImageHelpers

from ..models.classifier import ClassifierModelType
from ..models.image import ImageModelType
from .features_layers import (
    ExtractImageFeaturesLayer, 
    ExtractTextFeaturesLayer, 
    ExtractAudioFeaturesLayer, 
    ShareProjectionLayerV2
)
from .classifier import MLPClassifier, FastKANClassifier, MeanLogitsClassifier
from .gnn import MultiGATLayerV3
from .edge_weights import EdgeWeightsLayerV3


#=======================================================================================
class PipelineFull(nn.Module):
    MODAL_ORDER = ["images", "texts", "audios"] 

    def __init__(
        self, 
        image_model_name: ImageModelType, 
        out_feats_dim: int, 
        out_shared_dim: int, 
        num_classes: int, 
        classifier: ClassifierModelType
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        self.features_layers = nn.ModuleDict({
            "images": ExtractImageFeaturesLayer(image_model_name, in_channels=3, out_dim=out_feats_dim),
            "texts": ExtractTextFeaturesLayer(in_dim=384, out_dim=out_feats_dim),
            "audios": ExtractAudioFeaturesLayer(mfcc_dim=13, out_dim=out_feats_dim),
        })

        self.share_proj = ShareProjectionLayerV2(in_dim=out_feats_dim, out_dim=out_shared_dim, negative_slope=0.2)
        
        self.edge_weights_layer = EdgeWeightsLayerV3( 
            in_dim=out_shared_dim, 
            num_heads=8, 
            head_dim=64,
            dropout_rate=0.2 
        )
        
        self.multi_gnn_layer = MultiGATLayerV3(in_dim=out_shared_dim, out_dim=out_shared_dim, dropout_rate=0.2)

        if classifier == "mlp":
            self.classifier = MLPClassifier(in_dim=out_shared_dim, num_classes=num_classes)
        elif classifier == "fastkan":
            self.classifier = FastKANClassifier(in_dim=out_shared_dim, num_classes=num_classes)
        else:
            raise ValueError(f'classifier must be "mlp" or "fastkan", got "{classifier}"')
        
        self.to(self.device)

    def load(self, model_path: str) -> None:
        state_dict = torch.load(model_path, map_location=self.device)
        self.load_state_dict(state_dict)

    def _get_true_feat(self, graph_nodes: Dict[str, Any]) -> Dict[str, Tensor]:
        feats = {}
        
        images = graph_nodes.get("images_subgraph")
        if images:
            feats["images"] = torch.stack(images).float().to(self.device)
            
        audios = graph_nodes.get("audios_subgraph")
        if audios is not None and len(audios) > 0: 
             feats["audios"] = torch.from_numpy(np.array(audios)).float().to(self.device)
             
        texts = graph_nodes.get("texts_subgraph")
        if texts:
            feats["texts"] = (
                torch.stack([w["origin_word_embedding"] for w in texts], dim=0).to(self.device),
                torch.stack([w["origin_sent_embedding"] for w in texts], dim=0).to(self.device)
            )
            
        return feats

    def _extract_features(self, raw_feats: Dict[str, Tensor]) -> Dict[str, Tensor]:
        node_features = {}
        if "images" in raw_feats:
            node_features["images"] = self.features_layers["images"](raw_feats["images"])
        if "texts" in raw_feats:
            X_word, X_sent = raw_feats["texts"]
            node_features["texts"] = self.features_layers["texts"](X_word, X_sent)
        if "audios" in raw_feats:
            node_features["audios"] = self.features_layers["audios"](raw_feats["audios"])
        return node_features

    def _share_projection(self, node_features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        share_feats = {}
        for modal, feat_tensor in node_features.items():
            share_feats[modal] = self.share_proj(feat_tensor, modal)
        return share_feats

    def _build_conn_weights(self, all_feats_i: Tensor) -> Tensor:
        learned_weights_directed = self.edge_weights_layer(all_feats_i) 
        final_weights = (learned_weights_directed + learned_weights_directed.t()) / 2.0
        return final_weights
    
    def forward(self, graph_item: Dict[str, Any]) -> Tensor:
        graph_nodes = graph_item["graph_nodes"]
        
        raw_feats = self._get_true_feat(graph_nodes)
        node_features = self._extract_features(raw_feats)
        share_feats = self._share_projection(node_features)
        
        graph_item["share_feats"] = share_feats 
        all_features_list = []
        for modal in self.MODAL_ORDER:
            if modal in share_feats:
                all_features_list.append(share_feats[modal])
        
        all_features = torch.cat(all_features_list, dim=0)
        
        weights = self._build_conn_weights(all_features)
        graph_item["graph_weights"] = weights 
        gnn_output = self.multi_gnn_layer(all_features, weights)
        logits = self.classifier(gnn_output, weights)
        
        return logits


#=======================================================================================
class PipelineImageText(nn.Module):
    
    MODAL_ORDER = ["images", "texts"] 

    def __init__(self, image_model_name: ImageModelType, out_feats_dim: int, out_shared_dim: int, num_classes: int, classifier: str):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        self.features_layers = nn.ModuleDict({
            "images": ExtractImageFeaturesLayer(image_model_name, in_channels=3, out_dim=out_feats_dim),
            "texts": ExtractTextFeaturesLayer(in_dim=384, out_dim=out_feats_dim),
        })

        self.share_proj = ShareProjectionLayerV2(in_dim=out_feats_dim, out_dim=out_shared_dim, negative_slope=0.2)
        
        self.edge_weights_layer = EdgeWeightsLayerV3( 
            in_dim=out_shared_dim, 
            num_heads=8, 
            head_dim=64,
            dropout_rate=0.2 
        )
        
        self.multi_gnn_layer = MultiGATLayerV3(in_dim=out_shared_dim, out_dim=out_shared_dim, dropout_rate=0.2)

        if classifier == "mlp":
            self.classifier = MLPClassifier(in_dim=out_shared_dim, num_classes=num_classes)
        elif classifier == "fastkan":
            self.classifier = FastKANClassifier(in_dim=out_shared_dim, num_classes=num_classes)
        else:
            raise ValueError(f'classifier must be "mlp" or "fastkan", got "{classifier}"')
        
        self.to(self.device)

    def load(self, model_path: str) -> None:
        state_dict = torch.load(model_path, map_location=self.device)
        self.load_state_dict(state_dict)

    def _get_true_feat(self, graph_nodes: Dict[str, Any]) -> Dict[str, Tensor]:
        feats = {}
        
        images = graph_nodes.get("images_subgraph")
        if images:
            feats["images"] = torch.stack(images).float().to(self.device)
            
        texts = graph_nodes.get("texts_subgraph")
        if texts:
            feats["texts"] = (
                torch.stack([w["origin_word_embedding"] for w in texts], dim=0).to(self.device),
                torch.stack([w["origin_sent_embedding"] for w in texts], dim=0).to(self.device)
            )
            
        return feats

    def _extract_features(self, raw_feats: Dict[str, Tensor]) -> Dict[str, Tensor]:
        node_features = {}
        if "images" in raw_feats:
            node_features["images"] = self.features_layers["images"](raw_feats["images"])
        if "texts" in raw_feats:
            X_word, X_sent = raw_feats["texts"]
            node_features["texts"] = self.features_layers["texts"](X_word, X_sent)
        return node_features

    def _share_projection(self, node_features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        share_feats = {}
        for modal, feat_tensor in node_features.items():
            share_feats[modal] = self.share_proj(feat_tensor, modal)
        return share_feats

    def _build_conn_weights(self, all_feats_i: Tensor) -> Tensor:
        learned_weights_directed = self.edge_weights_layer(all_feats_i) 
        final_weights = (learned_weights_directed + learned_weights_directed.t()) / 2.0
        return final_weights
    
    def forward(self, graph_item: Dict[str, Any]) -> Tensor:
        graph_nodes = graph_item["graph_nodes"]
        
        raw_feats = self._get_true_feat(graph_nodes)
        node_features = self._extract_features(raw_feats)
        share_feats = self._share_projection(node_features)
        
        graph_item["share_feats"] = share_feats 
        all_features_list = []
        for modal in self.MODAL_ORDER:
            if modal in share_feats:
                all_features_list.append(share_feats[modal])
        
        all_features = torch.cat(all_features_list, dim=0)
        
        weights = self._build_conn_weights(all_features)
        graph_item["graph_weights"] = weights 
        gnn_output = self.multi_gnn_layer(all_features, weights)
        logits = self.classifier(gnn_output, weights)
        
        return logits
    

#=======================================================================================
class PipelineImage(nn.Module):
    def __init__(self, backbone_name: ImageModelType, num_classes: int):
        super().__init__()
        
        self.backbone, self.out_features = ImageHelpers.get_backbone(backbone_name, pretrained=True)
        self.classifier = MeanLogitsClassifier(self.out_features, num_classes)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def load(self, model_path: str) -> None:
        state_dict = torch.load(model_path, map_location=self.device)
        new_state_dict = {}
        
        for k, v in state_dict.items():
            if k.startswith("backbone.backbone."):
                new_key = k.replace("backbone.backbone.", "backbone.")
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v

        self.load_state_dict(new_state_dict)
                
    def forward(self, graph_item: Dict[str, Any]) -> Tensor:
        images_list = graph_item["graph_nodes"]["images_subgraph"]
        images = torch.stack(images_list).to(self.device)
        features = self.backbone(images)  # [Num_Images, Out_Features]
        logits = self.classifier(features) # [1, Num_Classes]
        
        return logits
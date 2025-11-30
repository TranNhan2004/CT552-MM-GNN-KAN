import torch
import torch.nn.functional as F

from numpy import ndarray
from torch import Tensor
from typing import Any, Dict, List
from ..models.image import ImageModelType
from ..ai.pipeline import PipelineV3

class ProcessService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline_models: Dict[ImageModelType, PipelineV3] = {
            "mobilenetv3large": PipelineV3(
                image_model_name="mobilenetv3large",
                out_feats_dim=448,
                out_shared_dim=512,
                num_classes=8,
                classifier="fastkan"
            ),
        }
        self.model_paths: Dict[ImageModelType, str] = {
            "mobilenetv3large": "/home/nhan/Workspace/All-Courses/Y4-S1/CT552/Application/backend/ai_models/pipeline/mobilenetv3large_fastkan.pt",
        }
        self.idx_to_label = {
            0: "Chợ nổi Cái Răng",
            1: "Đua bò Bảy Núi",
            2: "Hội Lim",
            3: "Lễ hội Nghinh Ông",
            4: "Nghề đan tre",
            5: "Nghề dệt chiếu",
            6: "Ok Om Bok",
            7: "Vía Bà Chúa Xứ Núi Sam",
        }
        
        for k, path in self.model_paths.items():
            state_dict = torch.load(path, map_location=self.device)
            migrated_state_dict = self._migrate_state_dict(state_dict)
            self.pipeline_models[k].load_state_dict(migrated_state_dict)
            self.pipeline_models[k].eval()
    
    def _migrate_state_dict(self, state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        new_state_dict = {}
        old_image_extractor_prefix = 'features_layers.images.backbone.'
        new_image_extractor_prefix = 'features_layers.images.backbone_model.backbone.'
        
        for key, value in state_dict.items():
            if key.startswith(new_image_extractor_prefix):
                new_state_dict[key] = value
            elif key.startswith(old_image_extractor_prefix):
                suffix = key[len(old_image_extractor_prefix):]
                new_key = new_image_extractor_prefix + suffix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        return new_state_dict

    def build_graph(
        self, 
        images_subgraph: List[Tensor], 
        texts_subgraph: List[Dict[str, Any]], 
        audios_subgraph: ndarray
    ) -> Dict[str, Any]:
        return {
            "graph_nodes": {
                "images_subgraph": images_subgraph,
                "texts_subgraph": texts_subgraph,
                "audios_subgraph": audios_subgraph,
            },
            "graph_weights": []
        }
    
    def predict(self, graph_item: Dict[str, Any], image_model_name: ImageModelType) -> Dict[str, Any]:
        with torch.no_grad():
            logits = self.pipeline_models[image_model_name](graph_item)
            pred = torch.argmax(logits, dim=1)
            weights = graph_item["graph_weights"].detach().cpu().numpy()
            label_idx = pred.item()
            label_name = self.idx_to_label[label_idx]
            
            probs = F.softmax(logits, dim=1)  
            max_prob = probs[0, label_idx].item()
        
        return {
            "weights": weights,
            "label_idx": label_idx,
            "label_name": label_name,
            "prob": max_prob
        }


        


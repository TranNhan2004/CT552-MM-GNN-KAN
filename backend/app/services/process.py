import torch
import torch.nn.functional as F

from numpy import ndarray
from torch import Tensor
from typing import Any, Dict, List, Literal
from ..models.image import ImageModelType
from ..ai.pipeline import PipelineFull, PipelineImage, PipelineImageText

class ProcessService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline_full_models: Dict[ImageModelType, PipelineFull] = {
            "resnet": PipelineFull(
                image_model_name="resnet",
                out_feats_dim=448,
                out_shared_dim=512,
                num_classes=8,
                classifier="fastkan"
            ),
            "regnet": PipelineFull(
                image_model_name="regnet",
                out_feats_dim=448,
                out_shared_dim=512,
                num_classes=8,
                classifier="mlp"
            ),
            "mobilenetv3large": PipelineFull(
                image_model_name="mobilenetv3large",
                out_feats_dim=448,
                out_shared_dim=512,
                num_classes=8,
                classifier="fastkan"
            ),
            "mobilenetv3small": PipelineFull(
                image_model_name="mobilenetv3small",
                out_feats_dim=448,
                out_shared_dim=512,
                num_classes=8,
                classifier="mlp"
            ),
            "densenet": PipelineFull(
                image_model_name="densenet",
                out_feats_dim=448,
                out_shared_dim=512,
                num_classes=8,
                classifier="fastkan"
            ),
            "shufflenet": PipelineFull(
                image_model_name="shufflenet",
                out_feats_dim=448,
                out_shared_dim=512,
                num_classes=8,
                classifier="mlp"
            ),
        }
        self.pipeline_full_model_paths: Dict[ImageModelType, str] = {
            "resnet": "ai_models/pipeline_full/fixed_multimodal_v3_resnet_fastkan_2025-12-06_07-51-50/best_model.pt",
            "regnet": "ai_models/pipeline_full/fixed_multimodal_v3_regnet_mlp_2025-12-06_08-48-01/best_model.pt",
            "mobilenetv3large": "ai_models/pipeline_full/fixed_multimodal_v3_mobilenetv3large_fastkan_2025-12-05_17-33-41/best_model.pt",
            "mobilenetv3small": "ai_models/pipeline_full/fixed_multimodal_v3_mobilenetv3small_mlp_2025-12-06_15-34-30/best_model.pt",
            "densenet": "ai_models/pipeline_full/fixed_multimodal_v3_densenet_fastkan_2025-12-05_14-02-15/best_model.pt",
            "shufflenet": "ai_models/pipeline_full/fixed_multimodal_v3_shufflenet_mlp_2025-12-05_22-48-24//best_model.pt",    
        }

        self.pipeline_image_text_models: Dict[ImageModelType, PipelineImageText] = {
            "resnet": PipelineImageText(
                image_model_name="resnet",
                out_feats_dim=448,
                out_shared_dim=512,
                num_classes=8,
                classifier="fastkan"
            ),
            "regnet": PipelineImageText(
                image_model_name="regnet",
                out_feats_dim=448,
                out_shared_dim=512,
                num_classes=8,
                classifier="mlp"
            ),
            "mobilenetv3large": PipelineImageText(
                image_model_name="mobilenetv3large",
                out_feats_dim=448,
                out_shared_dim=512,
                num_classes=8,
                classifier="mlp"
            ),
            "mobilenetv3small": PipelineImageText(
                image_model_name="mobilenetv3small",
                out_feats_dim=448,
                out_shared_dim=512,
                num_classes=8,
                classifier="mlp"
            ),
            "densenet": PipelineImageText(
                image_model_name="densenet",
                out_feats_dim=448,
                out_shared_dim=512,
                num_classes=8,
                classifier="fastkan"
            ),
            "shufflenet": PipelineImageText(
                image_model_name="shufflenet",
                out_feats_dim=448,
                out_shared_dim=512,
                num_classes=8,
                classifier="fastkan"
            ),
        }
        self.pipeline_image_text_model_paths: Dict[ImageModelType, str] = {
            "resnet": "ai_models/pipeline_img_txt/fixed_multimodal_v3_resnet_fastkan_2025-12-06_10-32-26/best_model.pt",
            "regnet": "ai_models/pipeline_img_txt/fixed_multimodal_v3_regnet_mlp_2025-12-06_17-29-32/best_model.pt",
            "mobilenetv3large": "ai_models/pipeline_img_txt/fixed_multimodal_v3_mobilenetv3large_mlp_2025-12-05_22-23-29/best_model.pt",
            "mobilenetv3small": "ai_models/pipeline_img_txt/fixed_multimodal_v3_mobilenetv3small_mlp_2025-12-05_23-21-59/best_model.pt",
            "densenet": "ai_models/pipeline_img_txt/fixed_multimodal_v3_densenet_fastkan_2025-12-06_20-24-49/best_model.pt",
            "shufflenet": "ai_models/pipeline_img_txt/fixed_multimodal_v3_shufflenet_fastkan_2025-12-06_00-51-23/best_model.pt",    
        }

        self.pipeline_image_models: Dict[ImageModelType, PipelineImage] = {
            "resnet": PipelineImage(
                backbone_name="resnet",
                num_classes=8
            ),
            "regnet": PipelineImage(
                backbone_name="regnet",
                num_classes=8
            ),
            "mobilenetv3large": PipelineImage(
                backbone_name="mobilenetv3large",
                num_classes=8
            ),
            "mobilenetv3small": PipelineImage(
                backbone_name="mobilenetv3small",
                num_classes=8
            ),
            "densenet": PipelineImage(
                backbone_name="densenet",
                num_classes=8
            ),
            "shufflenet": PipelineImage(
                backbone_name="shufflenet",
                num_classes=8
            ),
        }
        self.pipeline_image_model_paths: Dict[ImageModelType, str] = {
            "resnet": "ai_models/pipeline_img/resnet_2025-11-25_12-31-53/best_model.pt",
            "regnet": "ai_models/pipeline_img/regnet_2025-11-19_15-46-23/best_model.pt",
            "mobilenetv3large": "ai_models/pipeline_img/mobilenetv3large_2025-11-29_20-16-07/best_model.pt",
            "mobilenetv3small": "ai_models/pipeline_img/mobilenetv3small_2025-11-22_04-55-00/best_model.pt",
            "densenet": "ai_models/pipeline_img/densenet_2025-11-30_14-52-05/best_model.pt",
            "shufflenet": "ai_models/pipeline_img/shufflenet_2025-11-30_07-10-10/best_model.pt",    
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
        
        for (k, path_full), (_, path_image_text), (_, path_image) in zip(
            self.pipeline_full_model_paths.items(), 
            self.pipeline_image_text_model_paths.items(),
            self.pipeline_image_model_paths.items()
        ):
            self.pipeline_full_models[k].load(path_full)
            self.pipeline_full_models[k].eval()

            self.pipeline_image_text_models[k].load(path_image_text)
            self.pipeline_image_text_models[k].eval()
            
            self.pipeline_image_models[k].load(path_image)
            self.pipeline_image_models[k].eval()
    
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
    
    def predict(
        self, 
        graph_item: Dict[str, Any], 
        predict_type: Literal["full", "image_text", "image"], 
        image_model_name: ImageModelType
    ) -> Dict[str, Any]:
        with torch.no_grad():
            if predict_type == "full":
                logits = self.pipeline_full_models[image_model_name](graph_item)
                pred = torch.argmax(logits, dim=1)
                weights = graph_item["graph_weights"].detach().cpu().numpy()
                label_idx = pred.item()
                label_name = self.idx_to_label[label_idx]
                
                probs = F.softmax(logits, dim=1)  
                max_prob = probs[0, label_idx].item()
            
                return {
                    "weights": weights.tolist(),
                    "label_idx": label_idx,
                    "label_name": label_name,
                    "prob": max_prob
                }

            if predict_type == "image_text":
                logits = self.pipeline_image_text_models[image_model_name](graph_item)
                pred = torch.argmax(logits, dim=1)
                weights = graph_item["graph_weights"].detach().cpu().numpy()
                label_idx = pred.item()
                label_name = self.idx_to_label[label_idx]
                
                probs = F.softmax(logits, dim=1)  
                max_prob = probs[0, label_idx].item()
            
                return {
                    "weights": weights.tolist(),
                    "label_idx": label_idx,
                    "label_name": label_name,
                    "prob": max_prob
                }

            if predict_type == "image":
                logits = self.pipeline_image_models[image_model_name](graph_item)
                pred = torch.argmax(logits, dim=1)
                label_idx = pred.item()
                label_name = self.idx_to_label[label_idx]
                
                probs = F.softmax(logits, dim=1)  
                max_prob = probs[0, label_idx].item()
            
                return {
                    "label_idx": label_idx,
                    "label_name": label_name,
                    "prob": max_prob
                }

            raise ValueError(f'"predict_type must be "full", "image_text" or "image", got {predict_type}')

            


        


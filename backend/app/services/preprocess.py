from typing import Any, Dict, List, get_args
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose
from ..ai.audio_helpers import AudioHelpers
from ..ai.image_helpers import ImageHelpers
from ..ai.text_helpers import TextHelpers
from ..models.image import ImageModelType

class PreprocessService: 
    def __init__(self) -> None:
        self.images_transforms: Dict[ImageModelType, Compose] = {}
        for name in get_args(ImageModelType):
            self.images_transforms[name] = ImageHelpers.get_model_transforms(name)        

        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def preprocess_images(self, image_urls: List[str], image_model_name: ImageModelType) -> List[Tensor]:
        images_subgraph = []
        for img_path in image_urls:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.images_transforms[image_model_name](img)
                images_subgraph.append(img_tensor)
            except Exception as e:
                raise ValueError(f"Không thể load ảnh {img_path}: {e}")
        
        return images_subgraph
            
    def preprocess_texts(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return TextHelpers.compute_origin_embeddings(words, self.embedding_model)

    def preprocess_audios(self, audio_urls: List[str]) -> ndarray:
        audios_subgraph = []
        for aud_path in audio_urls:
            try:
                audio_mfcc= AudioHelpers.extract_mfcc_3d(aud_path)
                audios_subgraph.append(audio_mfcc)
            except Exception as e:
                raise ValueError(f"Không thể load âm thanh {aud_path}: {e}")

        return AudioHelpers.pad_to_max_shape(audios_subgraph)



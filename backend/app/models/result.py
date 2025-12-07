from pydantic import BaseModel
from typing import Dict, List, Optional

class ResultModel(BaseModel):
    id: Optional[int] = None

    image_urls: List[str] = []
    audio_urls: List[str] = []
    text: Optional[str] = None
    processed_texts: List[Dict] = []
    
    img_label_idx: Optional[int] = None
    img_label_name: Optional[str] = None
    img_prob: Optional[float] = None

    img_txt_label_idx: Optional[int] = None
    img_txt_label_name: Optional[str] = None
    img_txt_prob: Optional[float] = None
    img_txt_weights: List[List[float]] = []

    full_label_idx: Optional[int] = None
    full_label_name: Optional[str] = None
    full_prob: Optional[float] = None
    full_weights: List[List[float]] = []
    
    class Config:
        from_attributes = True
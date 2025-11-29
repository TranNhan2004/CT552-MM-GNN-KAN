from pydantic import BaseModel
from typing import Dict, List, Optional

class ResultModel(BaseModel):
    id: Optional[int] = None
    image_urls: List[str] = []
    audio_urls: List[str] = []
    text: Optional[str] = None
    processed_texts: List[Dict] = []
    label_idx: Optional[int] = None
    label_name: Optional[str] = None
    prob: Optional[float] = None
    weights: List[List[float]] = []

    class Config:
        orm_mode = True

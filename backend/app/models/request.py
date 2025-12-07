from pydantic import BaseModel

from .image import ImageModelType


class PredictReq(BaseModel):
    id: int
    model: ImageModelType
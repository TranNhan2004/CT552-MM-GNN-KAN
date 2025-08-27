from pydantic import BaseModel
from typing import List

class ResultResponse(BaseModel):
    text: str
    files: List[str]
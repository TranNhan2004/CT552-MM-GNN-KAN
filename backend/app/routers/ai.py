from fastapi import APIRouter, UploadFile, File, Form, Response
from typing import List
from app.entities import ResultResponse

router = APIRouter(
    prefix="/ai",   
    tags=["AI"]
)

@router.post("/predict")
async def predict(
    text: str = Form(...),
    files: List[UploadFile] = File([])  
):
    return ResultResponse(
        text=text,
        files=[file.filename for file in files]
    )

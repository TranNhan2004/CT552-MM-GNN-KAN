from typing import List
import uvicorn
import shutil, json, numpy as np

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from app.services.result import ResultService
from app.models.result import ResultModel
from app.services.upload import UploadService

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  
    allow_credentials=True,
    allow_methods=["*"],   
    allow_headers=["*"],   
)

DB_PATH = Path("results.db")
UPLOAD_ROOT_DIR = Path("uploads")

result_service = ResultService(DB_PATH)
upload_service = UploadService(UPLOAD_ROOT_DIR)

@app.post("/api/predict")
async def predict(files: List[UploadFile] = File(...), text: str = Form(...)):
    try:
        last_idx = result_service.find_last_id()
        upload_dir = f"{last_idx + 1}" if last_idx else "1"
        image_urls, full_text, audio_urls = upload_service.upload(files, upload_dir) 
        full_text += text if text else ""

        processed_text = [{"id": i+1, "word": w} for i,w in enumerate(text.split())]
        label_idx = 0
        label_name = "demo_label"
        prob = 0.9
        weights = np.random.rand(3,3).tolist()

        result_model = ResultModel(
            image_urls=image_urls,
            audio_urls=audio_urls,
            text=text,
            processed_text=processed_text,
            label_idx=label_idx,
            label_name=label_name,
            prob=prob,
            weights=weights
        )

        result_model = result_service.insert(result_model)
        return JSONResponse(result_model.model_dump_json())
    
    except Exception as e:
        return JSONResponse({"error": f"Lỗi khi upload video - {e}"}, status_code=400)

# --- GET result by id ---
@app.get("/api/result/{id}")
def get_result(id: int):
    record = result_service.find(id)
    if not record:
        return JSONResponse({"error": f"Không tìm thấy kết quả với ID={id}"}, status_code=404)
    return JSONResponse(record.model_dump_json())


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

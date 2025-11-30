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
from app.ai.text_helpers import TextHelpers
from app.services.preprocess import PreprocessService
from app.services.process import ProcessService
from app.models.image import ImageModelType

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  
    allow_credentials=True,
    allow_methods=["*"],   
    allow_headers=["*"],   
)

DB_PATH = Path("db/results.db")
UPLOAD_ROOT_DIR = Path("uploads")

NOT_USE_AUDIO = {
    'Chợ nổi Cái Răng',
    'Đua bò Bảy Núi',
    'Nghề đan tre',
    'Nghề dệt chiếu',
}

result_service = ResultService(DB_PATH)
upload_service = UploadService(UPLOAD_ROOT_DIR)
preprocess_service = PreprocessService()
process_service = ProcessService()

@app.post("/api/predict")
async def predict(
    files: List[UploadFile] = File(...), 
    text: str = Form(...), 
    model_name: ImageModelType = Form(...)
):
    try:
        last_idx = result_service.find_last_id()
        upload_dir = f"{last_idx + 1}" if last_idx else "1"
        image_urls, full_text, audio_urls = upload_service.upload(files, upload_dir) 
        full_text += text if text else ""

        words = TextHelpers.get_words(full_text)
        
        images_subgraph = preprocess_service.preprocess_images(image_urls, model_name)
        texts_subgraph = preprocess_service.preprocess_texts(words)
        audios_subgraph = preprocess_service.preprocess_audios(audio_urls)

        only_image_graph = process_service.build_graph(images_subgraph, [], [])
        coarse_pred = process_service.predict(only_image_graph)

        if coarse_pred["label_name"] in NOT_USE_AUDIO:
            upload_service.clean_audio(upload_dir)
            audio_urls = []
            audios_subgraph = []
            graph = process_service.build_graph(images_subgraph, texts_subgraph, [])
            pred = process_service.predict(graph)
        else:
            graph = process_service.build_graph(images_subgraph, texts_subgraph, audios_subgraph)
            pred = process_service.predict(graph)


        result_model = ResultModel(
            image_urls=image_urls,
            audio_urls=audio_urls,
            text=full_text,
            processed_texts=words,
            label_idx=pred["label_idx"],
            label_name=pred["label_name"],
            prob=pred["prob"],
            weights=pred["weights"]
        )

        result_model = result_service.insert(result_model)
        return JSONResponse(result_model.model_dump_json())
    
    except Exception as e:
        return JSONResponse({"error": f"{e}"}, status_code=400)


@app.get("/api/result/{id}")
async def get_result(id: int):
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

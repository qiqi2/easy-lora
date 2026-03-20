"""
FastAPI 后端服务器
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import tempfile
import os

from easy_lora.core.gpu_detector import detect_gpu
from easy_lora.core.config_recommender import recommend_config
from easy_lora.dataset.processor import DatasetProcessor
from easy_lora.export.exporter import ModelExporter, get_supported_formats

app = FastAPI(title="EasyLoRA API", version="0.1.0")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    model: str
    dataset_path: str
    lora_rank: int
    batch_size: int
    learning_rate: float
    epochs: int
    output_dir: str


class ExportRequest(BaseModel):
    model_path: str
    model_name: str
    format: str
    system_prompt: Optional[str] = ""


@app.get("/")
async def root():
    return {"message": "EasyLoRA API", "version": "0.1.0"}


@app.get("/api/gpu")
async def get_gpu_info():
    """获取 GPU 信息"""
    gpus = detect_gpu()
    
    result = []
    for gpu in gpus:
        config = recommend_config(gpu["vram_gb"])
        result.append({
            **gpu,
            "recommended_config": config
        })
    
    return {"gpus": result}


@app.get("/api/config/recommend/{vram_gb}")
async def get_recommended_config(vram_gb: float):
    """根据显存推荐配置"""
    config = recommend_config(vram_gb)
    return {"config": config}


@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """上传数据集"""
    
    # 保存上传的文件
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        processor = DatasetProcessor(tmp_path)
        stats = processor.get_stats()
        format_info = processor.detect_format()
        
        return {
            "success": True,
            "stats": stats,
            "format": format_info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/api/dataset/convert")
async def convert_dataset(
    file: UploadFile = File(...),
    text_column: Optional[str] = None,
    output_format: str = "conversation"
):
    """转换数据集格式"""
    
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        processor = DatasetProcessor(tmp_path)
        processor.load()
        
        if output_format == "conversation":
            converted = processor.to_conversation_format(text_column)
        
        return {
            "success": True,
            "converted_count": len(converted),
            "sample": converted[:3] if converted else []
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/export/formats")
async def get_export_formats():
    """获取支持的导出格式"""
    return {"formats": get_supported_formats()}


@app.post("/api/export")
async def export_model(request: ExportRequest):
    """导出模型"""
    
    try:
        exporter = ModelExporter(request.model_path, request.output_dir or "./output")
        
        if request.format == "ollama":
            output_path = exporter.export_to_ollama(request.model_name, request.system_prompt)
        elif request.format == "lmstudio":
            output_path = exporter.export_to_lmstudio(request.model_name)
        elif request.format == "gguf":
            output_path = exporter.export_to_gguf()
        else:
            raise HTTPException(status_code=400, detail=f"不支持的格式: {request.format}")
        
        return {
            "success": True,
            "output_path": output_path
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

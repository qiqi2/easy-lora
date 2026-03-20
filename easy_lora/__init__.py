# EasyLoRA
# 消费级显卡 LLM 微调助手

from .core.gpu_detector import detect_gpu
from .core.config_recommender import recommend_config
from .dataset.processor import DatasetProcessor
from .export.exporter import ModelExporter

__version__ = "0.1.0"
__all__ = [
    "detect_gpu",
    "recommend_config", 
    "DatasetProcessor",
    "ModelExporter"
]

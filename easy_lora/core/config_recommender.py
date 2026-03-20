"""
配置推荐模块
根据 GPU 显存推荐最佳训练参数
"""

from typing import Dict


def recommend_config(vram_gb: float) -> Dict:
    """根据显存推荐配置"""
    
    # 配置文件
    configs = [
        # 24GB+ (RTX 4090, A100 40GB)
        {
            "vram_range": (24, 100),
            "model": "Qwen/Qwen2-14B-Instruct",
            "lora_rank": 64,
            "batch_size": 4,
            "learning_rate": 2e-4,
            "gradient_accumulation": 4,
            "max_seq_length": 4096,
        },
        # 16GB (RTX 4080, A5000)
        {
            "vram_range": (16, 24),
            "model": "Qwen/Qwen2-7B-Instruct",
            "lora_rank": 32,
            "batch_size": 2,
            "learning_rate": 3e-4,
            "gradient_accumulation": 4,
            "max_seq_length": 2048,
        },
        # 12GB (RTX 3060 Ti, RTX 4070)
        {
            "vram_range": (12, 16),
            "model": "Qwen/Qwen2-7B-Instruct",
            "lora_rank": 16,
            "batch_size": 2,
            "learning_rate": 3e-4,
            "gradient_accumulation": 4,
            "max_seq_length": 2048,
        },
        # 8GB (RTX 3060, RTX 4060 Ti)
        {
            "vram_range": (8, 12),
            "model": "Qwen/Qwen2-7B-Instruct",
            "lora_rank": 8,
            "batch_size": 1,
            "learning_rate": 5e-4,
            "gradient_accumulation": 8,
            "max_seq_length": 1024,
        },
        # 6GB (RTX 2060, GTX 1660)
        {
            "vram_range": (6, 8),
            "model": "Qwen/Qwen2-1.8B-Instruct",
            "lora_rank": 16,
            "batch_size": 1,
            "learning_rate": 5e-4,
            "gradient_accumulation": 8,
            "max_seq_length": 1024,
        },
    ]
    
    for config in configs:
        vram_min, vram_max = config["vram_range"]
        if vram_min <= vram_gb < vram_max:
            return config
    
    # 默认配置
    return {
        "model": "Qwen/Qwen2-0.5B-Instruct",
        "lora_rank": 8,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "gradient_accumulation": 16,
        "max_seq_length": 512,
    }


def get_all_gpu_configs() -> Dict:
    """获取所有支持 GPU 的推荐配置"""
    gpus = [
        {"name": "RTX 4090", "vram_gb": 24},
        {"name": "RTX 4080 SUPER", "vram_gb": 16},
        {"name": "RTX 4070 Ti SUPER", "vram_gb": 16},
        {"name": "RTX 4070", "vram_gb": 12},
        {"name": "RTX 3060 Ti", "vram_gb": 8},
        {"name": "RTX 3060", "vram_gb": 12},
        {"name": "RTX 4060 Ti", "vram_gb": 8},
        {"name": "GTX 1660 SUPER", "vram_gb": 6},
    ]
    
    result = {}
    for gpu in gpus:
        result[gpu["name"]] = recommend_config(gpu["vram_gb"])
    
    return result

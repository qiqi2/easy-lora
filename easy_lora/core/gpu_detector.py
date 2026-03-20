"""
GPU 检测模块
"""

import subprocess
import platform
from typing import List, Dict, Optional


def detect_gpu() -> List[Dict]:
    """检测 GPU 信息"""
    if platform.system() != "Linux" and platform.system() != "Windows":
        return []
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                vram_gb = float(parts[1]) / 1024  # MB -> GB
                compute_cap = parts[2]
                
                gpus.append({
                    "name": parts[0],
                    "vram_gb": vram_gb,
                    "compute_capability": compute_cap,
                    "architecture": get_architecture(compute_cap)
                })
        
        return gpus
        
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def get_architecture(compute_cap: str) -> str:
    """根据计算能力获取架构名称"""
    arch_map = {
        "8.6": "Ampere (RTX 30xx)",
        "8.9": "Ada Lovelace (RTX 40xx)",
        "9.0": "Hopper (H100)",
        "7.5": "Turing (RTX 20xx)",
        "6.1": "Pascal (GTX 10xx)",
    }
    return arch_map.get(compute_cap, f"Unknown ({compute_cap})")


def get_gpu_count() -> int:
    """获取 GPU 数量"""
    return len(detect_gpu())

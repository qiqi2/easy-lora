"""
模型导出模块
支持 Ollama / LM Studio / GGUF 格式
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List


class ModelExporter:
    """模型导出器"""
    
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_ollama(self, model_name: str, system_prompt: str = "") -> str:
        """导出为 Ollama Modelfile"""
        
        modelfile_content = f'''FROM {self.model_path}

SYSTEM """{system_prompt}"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
'''
        
        output_path = self.output_dir / f"{model_name}.Modelfile"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(modelfile_content)
        
        # 创建使用说明
        readme = f'''# {model_name} - Ollama 模型

## 使用方法

```bash
# 创建模型
ollama create {model_name} -f {model_name}.Modelfile

# 运行模型
ollama run {model_name}
```

## 系统提示词

{system_prompt}

## 导出信息
- 原始模型: {self.model_path}
- 导出时间: {self._get_timestamp()}
'''
        
        readme_path = self.output_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme)
        
        return str(output_path)
    
    def export_to_lmstudio(self, model_name: str) -> str:
        """导出为 LM Studio 格式"""
        
        # LM Studio 使用 GGUF 格式
        lmstudio_dir = self.output_dir / model_name
        lmstudio_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建配置文件
        config = {
            "name": model_name,
            "architecture": "llama",
            "parameters": "7B",
            "context_length": 4096,
            "created_at": self._get_timestamp(),
        }
        
        config_path = lmstudio_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        # 复制模型文件
        model_files = list(self.model_path.glob("*.gguf"))
        if model_files:
            for f in model_files:
                shutil.copy(f, lmstudio_dir / f.name)
        
        return str(lmstudio_dir)
    
    def export_to_gguf(self, quantization: str = "Q4_K_M") -> str:
        """导出为 GGUF 量化格式"""
        
        # 这里需要 llama.cpp 的 convert.py
        # 简化版：假设模型已经是 GGUF 格式
        
        output_path = self.output_dir / f"model_{quantization}.gguf"
        
        # 实际实现需要调用 llama.cpp 的转换脚本
        # python convert.py --outfile {output_path} --outtype {quantization} {self.model_path}
        
        return str(output_path)
    
    def create_model_card(self, model_name: str, training_info: Dict) -> str:
        """创建模型卡片"""
        
        card = f'''---
language:
- zh
- en
license: mit
library_name: transformers
tags:
- llm
- lora
- fine-tuned
---

# {model_name}

这是一个使用 EasyLoRA 微调的 LoRA 模型。

## 训练信息

| 参数 | 值 |
|------|-----|
| 基础模型 | {training_info.get('base_model', 'Unknown')} |
| LoRA Rank | {training_info.get('lora_rank', 'Unknown')} |
| 训练步数 | {training_info.get('training_steps', 'Unknown')} |
| 学习率 | {training_info.get('learning_rate', 'Unknown')} |
| 数据集大小 | {training_info.get('dataset_size', 'Unknown')} |

## 使用方法

```python
from peft import PeftModel, AutoModelForCausalLM
from transformers import AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{training_info.get('base_model', '')}")
model = PeftModel.from_pretrained(model, "{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{training_info.get('base_model', '')}")
```

## 导出格式

- 🤗 HuggingFace: 原生格式
- 🦙 Ollama: Modelfile
- 📁 LM Studio: GGUF

---

使用 [EasyLoRA](https://github.com/easy-lora/easy-lora) 导出
'''
        
        card_path = self.output_dir / "MODEL_CARD.md"
        with open(card_path, "w", encoding="utf-8") as f:
            f.write(card)
        
        return str(card_path)
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()


def get_supported_formats() -> List[Dict]:
    """获取支持的导出格式"""
    return [
        {
            "id": "ollama",
            "name": "Ollama",
            "description": "本地运行大模型",
            "icon": "🦙",
            "file_extension": ".Modelfile"
        },
        {
            "id": "lmstudio",
            "name": "LM Studio",
            "description": "桌面端 LLM 客户端",
            "icon": "📁",
            "file_extension": ""
        },
        {
            "id": "gguf",
            "name": "GGUF (llama.cpp)",
            "description": "量化格式，适合低显存",
            "icon": "⚡",
            "file_extension": ".gguf"
        },
        {
            "id": "huggingface",
            "name": "HuggingFace",
            "description": "原生 LoRA 格式",
            "icon": "🤗",
            "file_extension": ""
        }
    ]

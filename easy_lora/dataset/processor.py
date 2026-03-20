"""
数据集处理模块
支持 CSV/JSONL/对话格式
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datasets import Dataset


class DatasetProcessor:
    """数据集处理器"""
    
    SUPPORTED_FORMATS = [".jsonl", ".json", ".csv"]
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.format = self.file_path.suffix.lower()
        self.raw_data = []
        
    def load(self) -> List[Dict]:
        """加载数据集"""
        if self.format == ".jsonl":
            self.raw_data = self._load_jsonl()
        elif self.format == ".json":
            self.raw_data = self._load_json()
        elif self.format == ".csv":
            self.raw_data = self._load_csv()
        else:
            raise ValueError(f"不支持的格式: {self.format}")
        
        return self.raw_data
    
    def _load_jsonl(self) -> List[Dict]:
        """加载 JSONL 文件"""
        data = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def _load_json(self) -> List[Dict]:
        """加载 JSON 文件"""
        with open(self.file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            if isinstance(content, list):
                return content
            return [content]
    
    def _load_csv(self) -> List[Dict]:
        """加载 CSV 文件"""
        df = pd.read_csv(self.file_path)
        return df.to_dict("records")
    
    def detect_format(self) -> Dict:
        """检测数据集格式"""
        if not self.raw_data:
            self.load()
        
        first_item = self.raw_data[0] if self.raw_data else {}
        
        # 检测对话格式
        if "messages" in first_item:
            return {
                "type": "conversation",
                "description": "对话格式 (OpenAI style)",
                "sample": first_item
            }
        
        # 检测指令格式
        if "instruction" in first_item and "output" in first_item:
            return {
                "type": "instruction",
                "description": "指令微调格式 (Alpaca style)",
                "sample": first_item
            }
        
        # 检测简单问答格式
        if "input" in first_item and "output" in first_item:
            return {
                "type": "qa",
                "description": "问答格式",
                "sample": first_item
            }
        
        return {
            "type": "unknown",
            "description": "未知格式",
            "sample": first_item
        }
    
    def to_conversation_format(self, text_column: Optional[str] = None) -> List[Dict]:
        """转换为对话格式"""
        if not self.raw_data:
            self.load()
        
        conversations = []
        
        for item in self.raw_data:
            # 已经是对话格式
            if "messages" in item:
                conversations.append(item)
                continue
            
            # 指令格式
            if "instruction" in item:
                messages = [
                    {"role": "user", "content": item["instruction"]}
                ]
                if "input" in item and item["input"]:
                    messages[0]["content"] += f"\n\n{item['input']}"
                messages.append({"role": "assistant", "content": item["output"]})
                conversations.append({"messages": messages})
                continue
            
            # CSV 格式
            if text_column and text_column in item:
                conversations.append({
                    "messages": [
                        {"role": "user", "content": item[text_column]}
                    ]
                })
        
        return conversations
    
    def clean_chinese_text(self, text: str) -> str:
        """中文文本清洗"""
        # 去除多余空格
        text = " ".join(text.split())
        
        # 统一标点符号
        replacements = {
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "：": ":",
            "；": ";",
            """: '"',
            """: '"',
            "'": "'",
            "'": "'",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()
    
    def get_stats(self) -> Dict:
        """获取数据集统计信息"""
        if not self.raw_data:
            self.load()
        
        total_samples = len(self.raw_data)
        
        # 计算平均长度
        total_length = 0
        for item in self.raw_data:
            if "messages" in item:
                for msg in item["messages"]:
                    total_length += len(msg.get("content", ""))
            elif "instruction" in item:
                total_length += len(item.get("instruction", ""))
                total_length += len(item.get("output", ""))
        
        avg_length = total_length / total_samples if total_samples > 0 else 0
        
        return {
            "total_samples": total_samples,
            "avg_length": round(avg_length, 2),
            "format": self.format,
            "file_size_mb": round(self.file_path.stat().st_size / 1024 / 1024, 2)
        }

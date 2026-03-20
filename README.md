# 🤖 EasyLoRA - 消费级显卡 LLM 微调助手

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/electron-28+-green.svg" alt="Electron">
  <img src="https://img.shields.io/badge/license-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/github/stars/easy-lora/easy-lora?style=social" alt="Stars">
</p>

> 让微调像"修图"一样简单 🖼️ → 🤖

## ✨ 特性

- 🖱️ **拖拽式操作** - 拖拽上传数据集，支持 CSV/JSONL/对话格式
- 🧠 **智能参数推荐** - 基于显卡型号自动推荐最佳 LoRA 参数
- 📊 **实时监控** - 显存占用和训练进度实时显示
- 📦 **一键导出** - 导出为 Ollama / LM Studio 格式
- 🇨🇳 **中文优化** - 内置中文数据集清洗模板

## 📸 截图


## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/easy-lora/easy-lora.git
cd easy-lora

# 安装 Python 依赖
pip install -r requirements.txt

# 运行
python -m easy_lora
```

### Docker 部署

```bash
docker-compose up -d
```

## 📖 使用指南

### 1. 上传数据集

支持格式：
- JSONL（每行一个 JSON）
- CSV（需指定文本列）
- 对话格式（role/content 对话数组）

### 2. 选择模型

推荐配置：

| 显卡型号 | VRAM | 推荐模型 | LoRA Rank | Batch Size |
|---------|------|---------|-----------|------------|
| RTX 3060 | 12GB | Qwen2-7B | 16 | 2 |
| RTX 3060 | 8GB | Qwen2-7B | 8 | 1 |
| RTX 4090 | 24GB | Qwen2-14B | 32 | 4 |
| A100 | 40GB | Qwen2-14B | 64 | 8 |

### 3. 开始训练

点击"开始训练"，实时查看：
- Loss 曲线
- 显存占用
- 预计剩余时间

### 4. 导出模型

支持导出格式：
- 🤗 HuggingFace GGUF
- 🦙 Ollama Modelfile
- 📁 LM Studio 格式

## 🏗️ 技术栈

- **前端**: Electron + React + TypeScript
- **后端**: Python 3.10+ + FastAPI
- **ML**: llama.cpp + peft + transformers
- **数据库**: SQLite

## 📁 项目结构

```
easy-lora/
├── easy_lora/           # 主应用
│   ├── ui/              # Electron 前端
│   ├── core/            # 核心训练逻辑
│   ├── dataset/         # 数据集处理
│   └── export/          # 模型导出
├── tests/               # 测试
├── docs/                # 文档
└── README.md
```

## 🤝 贡献

欢迎提交 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing`)
5. 打开 Pull Request

## 📝 许可证

MIT License - 查看 [LICENSE](LICENSE) 了解详情

---

⭐ 如果这个项目对你有帮助，请点个 Star！

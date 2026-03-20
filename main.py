#!/usr/bin/env python3
"""
EasyLoRA - 消费级显卡 LLM 微调助手
主入口文件
"""

import sys
import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def print_banner():
    """打印启动横幅"""
    banner = Text()
    banner.append("🤖 ", style="bold cyan")
    banner.append("EasyLoRA", style="bold green")
    banner.append(" - 消费级显卡 LLM 微调助手\n", style="dim")
    banner.append("让微调像\"修图\"一样简单 🖼️ → 🤖", style="italic")
    console.print(Panel(banner, border_style="green"))


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """EasyLoRA CLI"""
    pass


@cli.command()
@click.option("--host", default="127.0.0.1", help="服务器地址")
@click.option("--port", default=8000, help="服务器端口")
@click.option("--reload", is_flag=True, help="开发模式热重载")
def server(host: str, port: int, reload: bool):
    """启动后端服务器"""
    print_banner()
    
    import uvicorn
    from easy_lora.server import app
    
    console.print(f"[green]✓[/green] 启动服务器: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=reload)


@cli.command()
def gui():
    """启动 GUI 界面"""
    print_banner()
    
    from easy_lora.ui.main import main
    main()


@cli.command()
@click.argument("dataset_path")
@click.option("--model", default="Qwen/Qwen2-7B-Instruct", help="基础模型")
@click.option("--output", default="./output", help="输出目录")
def train(dataset_path: str, model: str, output: str):
    """命令行训练模式"""
    print_banner()
    
    console.print(f"[blue]📂[/blue] 数据集: {dataset_path}")
    console.print(f"[blue]🤖[/blue] 模型: {model}")
    console.print(f"[blue]📦[/blue] 输出: {output}")
    
    # TODO: 实现训练逻辑
    console.print("[yellow]⚠[/yellow] 命令行训练模式开发中...")


@cli.command()
def detect():
    """检测 GPU 信息"""
    from easy_lora.core.gpu_detector import detect_gpu
    
    gpu_info = detect_gpu()
    
    console.print("\n[bold]GPU 检测信息:[/bold]\n")
    
    if not gpu_info:
        console.print("[red]✗[/red] 未检测到 NVIDIA GPU")
        return
    
    for i, gpu in enumerate(gpu_info):
        console.print(f"[green]GPU {i}:[/green] {gpu['name']}")
        console.print(f"  显存: {gpu['vram_gb']:.1f} GB")
        console.print(f"  架构: {gpu['architecture']}")
        console.print(f"  计算能力: {gpu['compute_capability']}")
        
        # 推荐配置
        from easy_lora.core.config_recommender import recommend_config
        config = recommend_config(gpu['vram_gb'])
        
        console.print(f"\n[blue]推荐配置:[/blue]")
        console.print(f"  模型: {config['model']}")
        console.print(f"  LoRA Rank: {config['lora_rank']}")
        console.print(f"  Batch Size: {config['batch_size']}")
        console.print(f"  学习率: {config['learning_rate']}")


if __name__ == "__main__":
    cli()

"""
CLI interface for pyllama
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt

console = Console()
app = typer.Typer(name="pyllama", help="Python wrapper for llama.cpp server")


@app.command()
def serve(
    model: str = typer.Argument(..., help="Model name or path to GGUF file"),
    quant: str = typer.Option("Q4_K_M", "-q", "--quant", help="Quantization type"),
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(8080, "-p", "--port", help="Server port"),
    ctx_size: int = typer.Option(4096, "-c", "--ctx-size", help="Context size"),
    source: str = typer.Option("auto", "-s", "--source", help="Model source (huggingface/modelscope/auto)"),
    auto: bool = typer.Option(True, "--auto/--no-auto", help="Auto-detect best device"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
):
    """Start the llama.cpp server with automatic configuration."""
    from pyllama.runner import AutoRunner
    
    runner = AutoRunner(auto_detect_device=auto)
    
    try:
        server = runner.server(
            model=model,
            quant=quant,
            source=source,
            ctx_size=ctx_size,
            port=port,
            verbose=verbose
        )
        
        console.print(Panel(
            f"[green]Server running at:[/green] http://{host}:{port}\n"
            f"[blue]OpenAI API:[/blue] http://{host}:{port}/v1\n"
            f"[yellow]Press Ctrl+C to stop[/yellow]",
            title="pyllama server"
        ))
        
        server.start(wait=False)
        while server.is_running():
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
        if runner._current_server:
            runner._current_server.stop()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def run(
    model: str = typer.Argument(..., help="Model name or path"),
    prompt: str = typer.Option("Hello!", "-p", "--prompt", help="Prompt to send"),
    quant: str = typer.Option("Q4_K_M", "-q", "--quant", help="Quantization"),
    tokens: int = typer.Option(256, "-n", "--tokens", help="Max tokens to generate"),
    ctx_size: int = typer.Option(4096, "-c", "--ctx-size", help="Context size"),
    source: str = typer.Option("auto", "-s", "--source", help="Model source"),
    chat: bool = typer.Option(True, "--chat/--completion", help="Chat or completion mode"),
):
    """Run inference with a model (auto-downloads if needed)."""
    from pyllama.runner import AutoRunner
    
    runner = AutoRunner()
    
    try:
        result = runner.run(
            model=model,
            prompt=prompt,
            quant=quant,
            source=source,
            ctx_size=ctx_size,
            n_tokens=tokens,
            chat=chat
        )
        console.print(result)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("chat")
def chat_cmd(
    model: str = typer.Argument(..., help="Model name or path"),
    quant: str = typer.Option("Q4_K_M", "-q", "--quant", help="Quantization"),
    ctx_size: int = typer.Option(4096, "-c", "--ctx-size", help="Context size"),
    source: str = typer.Option("auto", "-s", "--source", help="Model source"),
):
    """Interactive chat with a model."""
    from pyllama.runner import AutoRunner
    from pyllama.client import Client
    
    runner = AutoRunner()
    
    try:
        server = runner.server(model=model, quant=quant, source=source, ctx_size=ctx_size)
        server.start()
        
        console.print("[green]Chat ready! Type 'exit' or 'quit' to end.[/green]\n")
        
        messages = []
        
        while True:
            user_input = Prompt.ask("[bold blue]You[/bold blue]")
            
            if user_input.lower() in ("exit", "quit", "q"):
                break
            
            messages.append({"role": "user", "content": user_input})
            
            client = Client(server.base_url)
            response = client.chat.completions.create(
                model="llama",
                messages=messages,
                max_tokens=512
            )
            
            if response.choices:
                assistant_msg = response.choices[0].get("message", {}).get("content", "")
                console.print(f"[bold green]Assistant[/bold green]: {assistant_msg}")
                messages.append({"role": "assistant", "content": assistant_msg})
        
        server.stop()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def download(
    model: str = typer.Argument(..., help="Model name (e.g., llama-3.2-3b) or repo (user/model)"),
    quant: str = typer.Option("Q4_K_M", "-q", "--quant", help="Quantization type"),
    source: str = typer.Option("auto", "-s", "--source", help="Model source (huggingface/modelscope/auto)"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output directory"),
    list_files: bool = typer.Option(False, "-l", "--list", help="List available files"),
):
    """Download a model from HuggingFace or ModelScope."""
    from pyllama.models import ModelDownloader, ModelRegistry
    
    downloader = ModelDownloader(output)
    
    registry_info = ModelRegistry.get_model_info(model, source)
    
    if registry_info:
        repo = registry_info["repo"]
        console.print(f"[blue]Found in registry: {registry_info['description']}[/blue]")
    else:
        repo = model
    
    if list_files:
        files = downloader.list_files(repo, source)
        if not files:
            console.print("[red]No GGUF files found[/red]")
            return
        
        table = Table(title=f"Available files in {repo}")
        table.add_column("Filename", style="cyan")
        table.add_column("Size", style="yellow")
        table.add_column("Quant", style="green")
        
        for filename, size in files:
            size_str = f"{size / (1024**3):.2f} GB" if size > 0 else "unknown"
            quant_str = downloader._extract_quantization(filename)
            table.add_row(filename, size_str, quant_str)
        
        console.print(table)
        return
    
    try:
        path = downloader.download(repo, None if quant == "auto" else f"*{quant}*", source, output)
        console.print(f"[green]Downloaded to: {path}[/green]")
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def models(
    local: bool = typer.Option(False, "-l", "--local", help="Show local models only"),
    search: Optional[str] = typer.Option(None, "--search", help="Search for models"),
):
    """List available models."""
    from pyllama.models import ModelDownloader, ModelRegistry
    from pyllama.runner import AutoRunner
    
    if local:
        runner = AutoRunner()
        models_list = runner.list_local_models()
        
        if not models_list:
            console.print("[yellow]No local models found[/yellow]")
            return
        
        table = Table(title="Local Models")
        table.add_column("Name", style="cyan")
        table.add_column("Path", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("Quant", style="magenta")
        
        for m in models_list:
            table.add_row(
                m.name,
                str(m.local_path.parent),
                f"{m.size_gb:.2f} GB",
                m.quantization
            )
        
        console.print(table)
    elif search:
        downloader = ModelDownloader()
        results = downloader.search(search)
        
        if not results:
            console.print(f"[yellow]No models found for '{search}'[/yellow]")
            return
        
        table = Table(title=f"Search Results for '{search}'")
        table.add_column("Repo", style="cyan")
        table.add_column("File", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("Source", style="magenta")
        
        for m in results[:20]:
            table.add_row(
                m.repo,
                m.filename[:50] + "..." if len(m.filename) > 50 else m.filename,
                f"{m.size_gb:.2f} GB",
                m.source
            )
        
        console.print(table)
    else:
        runner = AutoRunner()
        runner.list_available_models()


@app.command()
def devices():
    """List available GPU devices and their capabilities."""
    from pyllama.device import DeviceDetector
    
    detector = DeviceDetector()
    detector.detect()
    detector.list_devices()


@app.command()
def config(
    model: str = typer.Argument(..., help="Model path to analyze"),
    ctx_size: int = typer.Option(4096, "-c", "--ctx-size", help="Desired context size"),
):
    """Show optimal configuration for a model."""
    from pyllama.runner import AutoRunner
    
    if not Path(model).exists():
        console.print(f"[red]Model not found: {model}[/red]")
        raise typer.Exit(1)
    
    runner = AutoRunner()
    config = runner.detect_config(model, ctx_size)
    
    console.print(Panel(
        f"[green]Model:[/green] {config.model_path}\n"
        f"[blue]Backend:[/blue] {config.device_config.backend.value}\n"
        f"[yellow]Device:[/yellow] {config.device_config.device}\n"
        f"[magenta]GPU Layers:[/magenta] {config.device_config.n_gpu_layers if config.device_config.n_gpu_layers >= 0 else 'all'}\n"
        f"[cyan]Recommended Context:[/cyan] {config.device_config.recommended_ctx}\n"
        f"[dim]{config.device_config.notes}[/dim]",
        title="Recommended Configuration"
    ))


@app.command("build")
def build_binary(
    backend: str = typer.Option("vulkan", "-b", "--backend", help="GPU backend (vulkan, cuda, rocm, cpu)"),
    jobs: int = typer.Option(8, "-j", "--jobs", help="Parallel jobs"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output directory for binaries"),
):
    """Build llama.cpp binaries from source."""
    console.print(f"[blue]Building llama.cpp with {backend} backend...[/blue]")
    
    llama_cpp_dir = Path(__file__).parent.parent.parent
    build_dir = llama_cpp_dir / "build"
    
    cmake_args = [
        "cmake", "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    
    if backend == "vulkan":
        cmake_args.append("-DGGML_VULKAN=ON")
    elif backend == "cuda":
        cmake_args.append("-DGGML_CUDA=ON")
    elif backend == "rocm":
        cmake_args.append("-DGGML_HIP=ON")
    
    console.print(f"[dim]Running: {' '.join(cmake_args)}[/dim]")
    subprocess.run(cmake_args, cwd=llama_cpp_dir, check=True)
    
    build_cmd = ["cmake", "--build", str(build_dir), "-j", str(jobs)]
    subprocess.run(build_cmd, cwd=llama_cpp_dir, check=True)
    
    if output:
        import shutil
        output.mkdir(parents=True, exist_ok=True)
        for exe in (build_dir / "bin").glob("llama-*.exe" if sys.platform == "win32" else "llama-*"):
            shutil.copy2(exe, output / exe.name)
        console.print(f"[green]Binaries copied to: {output}[/green]")
    else:
        console.print(f"[green]Build complete! Binaries at: {build_dir / 'bin'}[/green]")


@app.command("download-binaries")
def download_binaries(
    backend: str = typer.Option("auto", "-b", "--backend", help="GPU backend (auto, vulkan, cuda, metal, cpu)"),
    force: bool = typer.Option(False, "-f", "--force", help="Force re-download"),
):
    """Download pre-built llama.cpp binaries."""
    from pyllama.binaries import get_binary_manager
    
    manager = get_binary_manager()
    
    console.print(f"[blue]Downloading binaries for {manager.platform}...[/blue]")
    
    try:
        manager.download_binaries(backend=backend, force=force)
        
        binaries = manager.list_cached_binaries()
        console.print(f"\n[green]Installed binaries:[/green]")
        for b in binaries[:10]:
            console.print(f"  - {b}")
        if len(binaries) > 10:
            console.print(f"  ... and {len(binaries) - 10} more")
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("clear-cache")
def clear_cache(
    model: Optional[str] = typer.Argument(None, help="Specific model to remove"),
    binaries: bool = typer.Option(False, "--binaries", help="Clear binary cache"),
):
    """Clear model and/or binary cache."""
    if binaries or model is None:
        from pyllama.binaries import get_binary_manager
        manager = get_binary_manager()
        manager.clear_cache()
        console.print("[green]Binary cache cleared[/green]")
    
    if model or not binaries:
        from pyllama.models import ModelDownloader
        downloader = ModelDownloader()
        downloader.clear_cache(model)
        console.print("[green]Model cache cleared[/green]")
    console.print("[green]Cache cleared[/green]")


if __name__ == "__main__":
    app()
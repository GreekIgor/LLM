"""
Скрипт для запуска vLLM сервера
Используйте этот скрипт для развёртывания локального LLM сервера
"""

import argparse
import subprocess
import sys

def start_vllm_server(model: str, host: str = "0.0.0.0", port: int = 8000, 
                      device: str = "auto", dtype: str = "auto"):
    """
    Запускает vLLM сервер с заданными параметрами
    
    Args:
        model: имя модели из HuggingFace Hub
        host: хост для сервера
        port: порт для сервера
        device: устройство (cpu/cuda/auto)
        dtype: тип данных (float16/bfloat16/auto)
    """
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
    ]
    
    if device != "auto":
        cmd.extend(["--device", device])
    
    if dtype != "auto":
        cmd.extend(["--dtype", dtype])
    
    print("🚀 Запуск vLLM сервера...")
    print(f"📦 Модель: {model}")
    print(f"🌐 Адрес: http://{host}:{port}")
    print(f"💻 Устройство: {device}")
    print(f"🔢 Тип данных: {dtype}")
    print("\n" + "="*70)
    print("Команда:", " ".join(cmd))
    print("="*70 + "\n")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n⏹️  Сервер остановлен пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка при запуске сервера: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Запуск vLLM сервера для локального LLM инференса"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-1.3b",
        help="Имя модели из HuggingFace Hub (по умолчанию: facebook/opt-1.3b)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Хост для сервера (по умолчанию: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Порт для сервера (по умолчанию: 8000)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Устройство для инференса (по умолчанию: auto)"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Тип данных для модели (по умолчанию: auto)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🔧 vLLM Server Launcher")
    print("="*70 + "\n")
    
    start_vllm_server(
        model=args.model,
        host=args.host,
        port=args.port,
        device=args.device,
        dtype=args.dtype
    )

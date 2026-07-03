#!/bin/bash
# Запуск ТОЛЬКО vLLM сервера на хосте (Linux/Mac).
# MLflow, inference-сервис и Prometheus поднимаются через docker compose.
#
# Порядок работы:
#   1) bash start_vllm.sh               # этот скрипт — vLLM на :8000
#   2) docker compose up --build        # остальной стек

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 HW9: запуск vLLM сервера (хост)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Python окружение
echo ""
echo "📦 Проверка виртуального окружения..."
if [ ! -d ".venv" ]; then
    echo "⚠️  .venv не найден. Создаю..."
    python -m venv .venv
fi
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "❌ Не найден скрипт активации .venv"
    exit 1
fi

# 2. Зависимости
echo ""
echo "📥 Установка зависимостей (vllm и пр.)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements_hw9.txt
echo "✅ Зависимости готовы"

# 3. Запуск vLLM (в текущем терминале, Ctrl+C для остановки)
MODEL=${MODEL:-"facebook/opt-1.3b"}
DEVICE=${DEVICE:-"auto"}

echo ""
echo "🚀 Запуск vLLM..."
echo "   Модель:     $MODEL"
echo "   Устройство: $DEVICE"
echo "   API:        http://localhost:8000"
echo "   (первая загрузка модели может занять 5-15 минут)"
echo ""
echo "После готовности vLLM в отдельном терминале выполните:"
echo "   docker compose up --build"
echo ""

python start_vllm_server.py --model "$MODEL" --device "$DEVICE" --port 8000

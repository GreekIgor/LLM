# Запуск ТОЛЬКО vLLM сервера на хосте (Windows PowerShell).
# MLflow, inference-сервис и Prometheus поднимаются через docker compose.
#
# Порядок работы:
#   1) .\start_vllm.ps1                 # этот скрипт — vLLM на :8000
#   2) docker compose up --build        # остальной стек

$ErrorActionPreference = "Stop"

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "🤖 HW9: запуск vLLM сервера (хост)" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan

# 1. Python окружение
Write-Host ""
Write-Host "📦 Проверка виртуального окружения..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    Write-Host "⚠️  .venv не найден. Создаю..." -ForegroundColor Yellow
    python -m venv .venv
}
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .venv\Scripts\Activate.ps1
} else {
    Write-Host "❌ Не найден скрипт активации .venv" -ForegroundColor Red
    exit 1
}

# 2. Зависимости
Write-Host ""
Write-Host "📥 Установка зависимостей (vllm и пр.)..." -ForegroundColor Yellow
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements_hw9.txt
Write-Host "✅ Зависимости готовы" -ForegroundColor Green

# 3. Запуск vLLM (в текущем терминале, Ctrl+C для остановки)
$model = if ($env:MODEL) { $env:MODEL } else { "facebook/opt-1.3b" }
$device = if ($env:DEVICE) { $env:DEVICE } else { "auto" }

Write-Host ""
Write-Host "🚀 Запуск vLLM..." -ForegroundColor Yellow
Write-Host "   Модель:     $model" -ForegroundColor Cyan
Write-Host "   Устройство: $device" -ForegroundColor Cyan
Write-Host "   API:        http://localhost:8000" -ForegroundColor Cyan
Write-Host "   (первая загрузка модели может занять 5-15 минут)" -ForegroundColor Yellow
Write-Host ""
Write-Host "После готовности vLLM в отдельном терминале выполните:" -ForegroundColor Green
Write-Host "   docker compose up --build" -ForegroundColor White
Write-Host ""

python start_vllm_server.py --model $model --device $device --port 8000

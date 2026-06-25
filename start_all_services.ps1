# Скрипт для запуска всех компонентов HW9 (Windows PowerShell)
# Основано на https://github.com/Ilia2704/llm_mlflow

$ErrorActionPreference = "Stop"

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "🚀 HW9: vLLM + MLflow Setup Script" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan

# 1. Проверка Python окружения
Write-Host ""
Write-Host "📦 Step 1: Checking Python environment..." -ForegroundColor Yellow

if (-not (Test-Path ".venv")) {
    Write-Host "⚠️  Virtual environment not found. Creating..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✅ Virtual environment exists" -ForegroundColor Green
}

# Активация venv
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .venv\Scripts\Activate.ps1
} else {
    Write-Host "❌ Cannot find activation script" -ForegroundColor Red
    exit 1
}

# 2. Установка зависимостей
Write-Host ""
Write-Host "📥 Step 2: Installing dependencies..." -ForegroundColor Yellow
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements_hw9.txt
Write-Host "✅ Dependencies installed" -ForegroundColor Green

# 3. Запуск MLflow
Write-Host ""
Write-Host "📊 Step 3: Starting MLflow tracking server..." -ForegroundColor Yellow

$mlflowJob = Start-Job -ScriptBlock {
    param($venvPath)
    & "$venvPath\Scripts\python.exe" -m mlflow server `
        --host 0.0.0.0 `
        --port 5000 `
        --backend-store-uri sqlite:///mlflow.db `
        --default-artifact-root ./mlruns
} -ArgumentList (Get-Location).Path + "\.venv"

Write-Host "✅ MLflow started (Job ID: $($mlflowJob.Id))" -ForegroundColor Green
Write-Host "   UI: http://localhost:5000" -ForegroundColor Cyan

# Ждём запуска MLflow
Write-Host "   Waiting for MLflow to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

for ($i = 1; $i -le 10; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5000" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        Write-Host "   ✅ MLflow is ready!" -ForegroundColor Green
        break
    } catch {
        Write-Host "   ⏳ Waiting... ($i/10)" -ForegroundColor Yellow
        Start-Sleep -Seconds 2
    }
}

# 4. Запуск vLLM
Write-Host ""
Write-Host "🤖 Step 4: Starting vLLM server..." -ForegroundColor Yellow

$model = if ($env:MODEL) { $env:MODEL } else { "facebook/opt-1.3b" }
$device = if ($env:DEVICE) { $env:DEVICE } else { "auto" }

Write-Host "   Model: $model" -ForegroundColor Cyan
Write-Host "   Device: $device" -ForegroundColor Cyan

$vllmJob = Start-Job -ScriptBlock {
    param($venvPath, $model, $device)
    & "$venvPath\Scripts\python.exe" start_vllm_server.py --model $model --device $device --port 8000
} -ArgumentList (Get-Location).Path + "\.venv", $model, $device

Write-Host "✅ vLLM starting (Job ID: $($vllmJob.Id))" -ForegroundColor Green
Write-Host "   API: http://localhost:8000" -ForegroundColor Cyan

# Ждём запуска vLLM
Write-Host "   ⏳ Waiting for vLLM to load model (this may take 5-15 minutes)..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

for ($i = 1; $i -le 60; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        Write-Host "   ✅ vLLM is ready!" -ForegroundColor Green
        break
    } catch {
        Write-Host "   ⏳ Loading model... ($i/60)" -ForegroundColor Yellow
        Start-Sleep -Seconds 10
    }
}

# 5. Тестирование
Write-Host ""
Write-Host "🧪 Step 5: Testing services..." -ForegroundColor Yellow
python test_vllm_server.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ All tests passed!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Some tests failed. Check logs above." -ForegroundColor Yellow
}

# 6. Запуск Inference Service
Write-Host ""
Write-Host "🌐 Step 6: Starting FastAPI Inference Service..." -ForegroundColor Yellow

$inferenceJob = Start-Job -ScriptBlock {
    param($venvPath)
    & "$venvPath\Scripts\python.exe" inference_service.py
} -ArgumentList (Get-Location).Path + "\.venv"

Write-Host "✅ Inference service started (Job ID: $($inferenceJob.Id))" -ForegroundColor Green
Write-Host "   API: http://localhost:8080" -ForegroundColor Cyan
Write-Host "   Docs: http://localhost:8080/docs" -ForegroundColor Cyan
Write-Host "   Metrics: http://localhost:8080/metrics" -ForegroundColor Cyan

# Ждём запуска Inference Service
Start-Sleep -Seconds 3

# 7. Вывод информации
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host "✅ All services are running!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host ""
Write-Host "📊 MLflow UI:           http://localhost:5000" -ForegroundColor Cyan
Write-Host "🤖 vLLM API:            http://localhost:8000" -ForegroundColor Cyan
Write-Host "🌐 Inference Service:   http://localhost:8080" -ForegroundColor Cyan
Write-Host "📖 API Docs:            http://localhost:8080/docs" -ForegroundColor Cyan
Write-Host "📈 Prometheus Metrics:  http://localhost:8080/metrics" -ForegroundColor Cyan
Write-Host ""
Write-Host "Job IDs:" -ForegroundColor Yellow
Write-Host "  MLflow:    $($mlflowJob.Id)" -ForegroundColor White
Write-Host "  vLLM:      $($vllmJob.Id)" -ForegroundColor White
Write-Host "  Inference: $($inferenceJob.Id)" -ForegroundColor White
Write-Host ""
Write-Host "To view job output:" -ForegroundColor Yellow
Write-Host "  Get-Job | Receive-Job -Keep" -ForegroundColor White
Write-Host ""
Write-Host "To stop all services:" -ForegroundColor Yellow
Write-Host "  Get-Job | Stop-Job; Get-Job | Remove-Job" -ForegroundColor White
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "🎓 Ready for homework! Open hw9_vllm_mlflow.ipynb" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop all services..." -ForegroundColor Yellow

# Ожидание (держим скрипт запущенным)
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    Write-Host ""
    Write-Host "Stopping all services..." -ForegroundColor Yellow
    Get-Job | Stop-Job
    Get-Job | Remove-Job
    Write-Host "✅ All services stopped" -ForegroundColor Green
}

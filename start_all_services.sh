#!/bin/bash
# Скрипт для запуска всех компонентов HW9
# Основано на https://github.com/Ilia2704/llm_mlflow

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 HW9: vLLM + MLflow Setup Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Проверка Python окружения
echo ""
echo "📦 Step 1: Checking Python environment..."
if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found. Creating..."
    python -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment exists"
fi

# Активация venv
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "❌ Cannot find activation script"
    exit 1
fi

# 2. Установка зависимостей
echo ""
echo "📥 Step 2: Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements_hw9.txt
echo "✅ Dependencies installed"

# 3. Запуск MLflow
echo ""
echo "📊 Step 3: Starting MLflow tracking server..."
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns &
MLFLOW_PID=$!
echo "✅ MLflow started (PID: $MLFLOW_PID)"
echo "   UI: http://localhost:5000"

# Ждём запуска MLflow
echo "   Waiting for MLflow to start..."
sleep 5
for i in {1..10}; do
    if curl -s http://localhost:5000 > /dev/null 2>&1; then
        echo "   ✅ MLflow is ready!"
        break
    fi
    echo "   ⏳ Waiting... ($i/10)"
    sleep 2
done

# 4. Запуск vLLM
echo ""
echo "🤖 Step 4: Starting vLLM server..."
MODEL=${MODEL:-"facebook/opt-1.3b"}
DEVICE=${DEVICE:-"auto"}

echo "   Model: $MODEL"
echo "   Device: $DEVICE"

python start_vllm_server.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --port 8000 &
VLLM_PID=$!
echo "✅ vLLM starting (PID: $VLLM_PID)"
echo "   API: http://localhost:8000"

# Ждём запуска vLLM (может занять время на загрузку модели)
echo "   ⏳ Waiting for vLLM to load model (this may take 5-15 minutes)..."
sleep 10
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "   ✅ vLLM is ready!"
        break
    fi
    echo "   ⏳ Loading model... ($i/60)"
    sleep 10
done

# 5. Тестирование
echo ""
echo "🧪 Step 5: Testing services..."
python test_vllm_server.py
if [ $? -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "⚠️  Some tests failed. Check logs above."
fi

# 6. Запуск Inference Service
echo ""
echo "🌐 Step 6: Starting FastAPI Inference Service..."
python inference_service.py &
INFERENCE_PID=$!
echo "✅ Inference service started (PID: $INFERENCE_PID)"
echo "   API: http://localhost:8080"
echo "   Docs: http://localhost:8080/docs"
echo "   Metrics: http://localhost:8080/metrics"

# 7. Вывод информации
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All services are running!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 MLflow UI:           http://localhost:5000"
echo "🤖 vLLM API:            http://localhost:8000"
echo "🌐 Inference Service:   http://localhost:8080"
echo "📖 API Docs:            http://localhost:8080/docs"
echo "📈 Prometheus Metrics:  http://localhost:8080/metrics"
echo ""
echo "Process IDs:"
echo "  MLflow:    $MLFLOW_PID"
echo "  vLLM:      $VLLM_PID"
echo "  Inference: $INFERENCE_PID"
echo ""
echo "To stop all services, run:"
echo "  kill $MLFLOW_PID $VLLM_PID $INFERENCE_PID"
echo ""
echo "Or save PIDs to file:"
cat > .pids << EOF
MLFLOW_PID=$MLFLOW_PID
VLLM_PID=$VLLM_PID
INFERENCE_PID=$INFERENCE_PID
EOF
echo "  PIDs saved to .pids"
echo "  To stop: source .pids && kill \$MLFLOW_PID \$VLLM_PID \$INFERENCE_PID"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎓 Ready for homework! Open hw9_vllm_mlflow.ipynb"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Ожидание (чтобы скрипт не завершился)
wait

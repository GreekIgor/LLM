# Образ для FastAPI Inference Service (inference_service.py)
FROM python:3.11-slim

WORKDIR /app

# Ставим зависимости отдельным слоем — кэшируется, пока не меняется requirements
COPY requirements_inference.txt .
RUN pip install --no-cache-dir -r requirements_inference.txt

# Код сервиса
COPY inference_service.py .

EXPOSE 8080

CMD ["python", "inference_service.py"]

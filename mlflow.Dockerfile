# Образ для MLflow tracking server
FROM python:3.11-slim

RUN pip install --no-cache-dir "mlflow>=2.9.0"

WORKDIR /mlflow

EXPOSE 5000

# sqlite backend + serve-artifacts, чтобы клиент мог логировать артефакты через сервер.
# --allowed-hosts=* снимает защиту от DNS rebinding: inference ходит по хосту "mlflow:5000",
# который иначе отклоняется. Для локального стенда это безопасно.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/mlruns", \
     "--serve-artifacts", \
     "--allowed-hosts", "*"]

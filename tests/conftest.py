"""Общие фикстуры и настройка путей для pytest (ДЗ-19)."""
import sys
from pathlib import Path

# Корень проекта в sys.path, чтобы импортировать rag_app/config/corpus из tests/.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wan2.1 Ultimate UI - Полноценный интерфейс для генерации видео и изображений
================================================================================
Поддерживает:
  - Text-to-Video (T2V) - генерация видео из текста
  - Image-to-Video (I2V) - генерация видео из изображения
  - Text-to-Image (T2I) - генерация изображений из текста

Особенности:
  - Автоматическая установка всех зависимостей
  - Автоматический поиск свободного порта
  - Скачивание моделей через HuggingFace
  - Поиск и установка LoRA
  - Прогресс-бары для всех операций
  - Автооткрытие браузера
  - Поддержка метаданных LoRA (триггерные слова)
  - Предпросмотр LoRA (preview.png, preview.jpg, preview.webp)
  - Поддержка CivitAI для получения превью (без NSFW фильтра)
  - Фильтрация LoRA по типу задачи

Для вашего железа (RTX 3060 Ti 8GB):
  - T2V 1.3B - РАБОТАЕТ
  - I2V 14B FP8 - РАБОТАЕТ (15-20 минут)
  - T2V 14B - НЕ РАБОТАЕТ (требует 24GB VRAM)

Запуск: python wan2.1_ultimate_ui.py
"""

import subprocess
import sys
import os
import tempfile
import shutil
import webbrowser
import threading
import requests
import json
import time
import socket
import argparse
import re
from pathlib import Path
from PIL import Image
import io

# ============================================================
# АВТОМАТИЧЕСКАЯ УСТАНОВКА КОМПОНЕНТОВ
# ============================================================

def check_and_install_python_packages():
    """Проверяет и устанавливает необходимые Python пакеты"""
    required_packages = [
        "gradio",
        "requests",
        "tqdm",
        "huggingface_hub",
        "safetensors",
        "pillow",
    ]
    
    # Проверяем torch отдельно (с поддержкой CUDA)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch уже установлен с CUDA {torch.version.cuda}")
        else:
            print("⚠️ PyTorch установлен, но CUDA не доступна")
    except ImportError:
        print("📦 Устанавливаю PyTorch с поддержкой CUDA...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} уже установлен")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} не найден")
    
    if missing_packages:
        print(f"\n📦 Устанавливаю недостающие пакеты: {', '.join(missing_packages)}")
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("✅ Все пакеты установлены!")
    
    return True

def clone_wan_repository():
    """Клонирует репозиторий Wan2.1 если он отсутствует"""
    if Path("./Wan2.1/generate.py").exists():
        print("✅ Репозиторий Wan2.1 уже существует")
        return True
    
    print("📦 Клонирую репозиторий Wan2.1...")
    try:
        subprocess.check_call([
            "git", "clone", "https://github.com/Wan-Video/Wan2.1.git"
        ])
        print("✅ Репозиторий успешно склонирован")
        return True
    except Exception as e:
        print(f"❌ Ошибка клонирования: {e}")
        print("⚠️ Скачайте репозиторий вручную: git clone https://github.com/Wan-Video/Wan2.1.git")
        return False

def setup_wan_environment():
    """Устанавливает зависимости Wan2.1"""
    if not Path("./Wan2.1/requirements.txt").exists():
        print("❌ requirements.txt не найден")
        return False
    
    print("📦 Устанавливаю зависимости Wan2.1...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "./Wan2.1/requirements.txt"
        ])
        print("✅ Зависимости Wan2.1 установлены")
        return True
    except Exception as e:
        print(f"❌ Ошибка установки зависимостей: {e}")
        return False

def auto_install_all():
    """Автоматическая установка всех компонентов"""
    print("=" * 70)
    print("🚀 Wan2.1 Ultimate UI - Автоматическая установка")
    print("=" * 70)
    
    # Шаг 1: Установка Python пакетов
    print("\n[1/4] Проверка Python пакетов...")
    check_and_install_python_packages()
    
    # Шаг 2: Клонирование репозитория
    print("\n[2/4] Проверка репозитория Wan2.1...")
    clone_wan_repository()
    
    # Шаг 3: Установка зависимостей Wan2.1
    print("\n[3/4] Установка зависимостей Wan2.1...")
    setup_wan_environment()
    
    # Шаг 4: Проверка CUDA
    print("\n[4/4] Проверка CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA доступна. Версия: {torch.version.cuda}")
            print(f"   Видеокарта: {torch.cuda.get_device_name(0)}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   VRAM: {vram_gb:.1f} GB")
            if vram_gb < 10:
                print("   ⚠️ У вас 8GB VRAM. Работают модели: T2V 1.3B, I2V 14B FP8")
        else:
            print("⚠️ CUDA не обнаружена. Будет использован CPU (очень медленно)")
    except:
        print("⚠️ Не удалось проверить CUDA")
    
    print("\n" + "=" * 70)
    print("✅ Установка завершена! Запускаю интерфейс...")
    print("=" * 70 + "\n")

# Запускаем автоматическую установку ПЕРЕД импортом остальных модулей
auto_install_all()

# ============================================================
# ИМПОРТЫ ПОСЛЕ УСТАНОВКИ
# ============================================================

import gradio as gr
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm
import safetensors.torch as sf

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

# Определяем пути (работаем внутри папки Wan2.1)
WAN_DIR = Path("./Wan2.1")
OUTPUT_DIR = WAN_DIR / "outputs"
MODELS_DIR = WAN_DIR / "models"
LORA_DIR = WAN_DIR / "models" / "lora"
GENERATE_SCRIPT = WAN_DIR / "generate.py"

# Создаём папки
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)
LORA_DIR.mkdir(exist_ok=True, parents=True)

# API для поиска
api = HfApi()

# Кэш для поиска
SEARCH_CACHE_FILE = Path("./search_cache.json")

# Ссылки на модели (HuggingFace)
MODELS = {
    "t2v_1.3b": {
        "name": "Text-to-Video 1.3B",
        "repo_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "size_gb": 2.8,
        "required_vram": 8,
        "works_on_8gb": True
    },
    "t2v_14b": {
        "name": "Text-to-Video 14B",
        "repo_id": "Wan-AI/Wan2.1-T2V-14B",
        "size_gb": 16.0,
        "required_vram": 24,
        "works_on_8gb": False
    },
    "i2v_14b_fp8": {
        "name": "Image-to-Video 14B (FP8)",
        "repo_id": "Wan-AI/Wan2.1-I2V-14B-FP8",
        "size_gb": 16.0,
        "required_vram": 12,
        "works_on_8gb": True
    }
}

# ============================================================
# ФУНКЦИЯ ПОИСКА СВОБОДНОГО ПОРТА
# ============================================================

def find_free_port(start_port=7860, max_attempts=10):
    """Ищет свободный порт, начиная с start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            print(f"⚠️ Порт {port} занят, пробую следующий...")
            continue
    
    print(f"❌ Не удалось найти свободный порт в диапазоне {start_port}-{start_port + max_attempts - 1}")
    return None

# ============================================================
# ФУНКЦИИ ДЛЯ РАБОТЫ C CIVITAI API
# ============================================================

def extract_civitai_model_id(civitai_url):
    """Извлекает ID модели из URL CivitAI"""
    if not civitai_url:
        return None
    match = re.search(r'/models/(\d+)', civitai_url)
    return match.group(1) if match else None

def get_civitai_model_info(model_id):
    """Получает информацию о модели с CivitAI"""
    url = f"https://civitai.com/api/v1/models/{model_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Ошибка получения модели с CivitAI: {e}")
        return None

def get_civitai_model_images(model_id, limit=5):
    """Получает список изображений модели с CivitAI (БЕЗ NSFW ФИЛЬТРА)"""
    url = f"https://civitai.com/api/v1/images"
    params = {
        "modelId": model_id,
        "limit": limit
        # NSFW фильтр УБРАН - показываем всё
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        images = []
        for item in data.get("items", []):
            images.append({
                "url": item["url"],
                "width": item.get("width"),
                "height": item.get("height"),
                "nsfw": item.get("nsfw", False),
                "meta": item.get("meta", {})
            })
        return images
    except Exception as e:
        print(f"Ошибка получения изображений с CivitAI: {e}")
        return []

def download_image_from_url(url, save_path):
    """Скачивает изображение по URL и сохраняет локально"""
    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()
        
        # Определяем формат из URL или Content-Type
        content_type = response.headers.get('content-type', '')
        if 'png' in content_type or url.endswith('.png'):
            ext = '.png'
        elif 'jpeg' in content_type or 'jpg' in content_type or url.endswith('.jpg') or url.endswith('.jpeg'):
            ext = '.jpg'
        elif 'webp' in content_type or url.endswith('.webp'):
            ext = '.webp'
        else:
            ext = '.png'
        
        final_path = save_path.with_suffix(ext)
        
        with open(final_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return str(final_path)
    except Exception as e:
        print(f"Ошибка скачивания изображения: {e}")
        return None

# ============================================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С LORA (МЕТАДАННЫЕ, ПРЕВЬЮ, CIVITAI)
# ============================================================

def get_lora_metadata(lora_path):
    """Извлекает метаданные из LoRA файла"""
    lora_dir = Path(lora_path)
    safetensors_files = list(lora_dir.glob("*.safetensors"))
    if not safetensors_files:
        return {}
    
    try:
        metadata = sf.safe_open(safetensors_files[0], framework="pt").metadata()
        if metadata:
            return {
                "trigger_words": metadata.get("ss_trigger_words", ""),
                "base_model": metadata.get("ss_base_model_version", ""),
                "description": metadata.get("ss_description", ""),
                "author": metadata.get("ss_author", ""),
            }
    except Exception as e:
        print(f"Ошибка чтения метаданных: {e}")
    
    # Пробуем прочитать info.json
    info_file = lora_dir / "info.json"
    if info_file.exists():
        try:
            with open(info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    
    return {}

def get_lora_preview(lora_name):
    """Возвращает путь к превью-изображению LoRA (локальное или с CivitAI)"""
    lora_path = LORA_DIR / lora_name
    
    # 1. Сначала проверяем локальные превью
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        preview_path = lora_path / f"preview{ext}"
        if preview_path.exists():
            return str(preview_path)
    
    # 2. Проверяем info.json на наличие ссылки на CivitAI
    info_file = lora_path / "info.json"
    if info_file.exists():
        try:
            with open(info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
                civitai_url = info.get("civitai_url")
                if civitai_url:
                    model_id = extract_civitai_model_id(civitai_url)
                    if model_id:
                        # Получаем изображения с CivitAI
                        images = get_civitai_model_images(model_id, limit=1)
                        if images:
                            # Скачиваем первое изображение и сохраняем как preview
                            preview_url = images[0]["url"]
                            local_preview = lora_path / "preview.png"
                            result = download_image_from_url(preview_url, local_preview)
                            if result:
                                return result
        except Exception as e:
            print(f"Ошибка обработки info.json для {lora_name}: {e}")
    
    return None

def get_lora_trigger_words(lora_name):
    """Возвращает триггерные слова для LoRA"""
    if not lora_name or lora_name == "Без LoRA":
        return ""
    lora_path = LORA_DIR / lora_name
    metadata = get_lora_metadata(lora_path)
    return metadata.get("trigger_words", "")

def get_lora_info_text(lora_name):
    """Возвращает форматированную информацию о LoRA для отображения"""
    if not lora_name or lora_name == "Без LoRA":
        return "Выберите LoRA для просмотра информации"
    
    lora_path = LORA_DIR / lora_name
    metadata = get_lora_metadata(lora_path)
    
    if not metadata:
        return "Информация о LoRA не найдена"
    
    info_lines = []
    if metadata.get("author"):
        info_lines.append(f"👤 Автор: {metadata['author']}")
    if metadata.get("trigger_words"):
        info_lines.append(f"🔑 Триггерные слова: `{metadata['trigger_words']}`")
    if metadata.get("base_model"):
        info_lines.append(f"📦 Базовая модель: {metadata['base_model']}")
    if metadata.get("description"):
        info_lines.append(f"📝 Описание: {metadata['description']}")
    if metadata.get("civitai_url"):
        info_lines.append(f"🔗 CivitAI: {metadata['civitai_url']}")
    
    return "\n".join(info_lines) if info_lines else "Дополнительная информация отсутствует"

def get_loras_by_task(task_type):
    """Возвращает LoRA, подходящие для указанной задачи"""
    all_loras = get_installed_loras()
    if not all_loras:
        return ["Без LoRA"]
    
    filtered = ["Без LoRA"]
    for lora in all_loras:
        lora_path = LORA_DIR / lora["name"]
        metadata = get_lora_metadata(lora_path)
        
        # Если нет метаданных, показываем LoRA (пользователь сам разберётся)
        if not metadata:
            filtered.append(lora["name"])
            continue
        
        # Фильтруем по типу задачи
        base_model = metadata.get("base_model", "")
        if task_type == "T2V" and ("1.3B" in base_model or "T2V" in base_model):
            filtered.append(lora["name"])
        elif task_type == "I2V" and ("14B" in base_model or "I2V" in base_model):
            filtered.append(lora["name"])
        elif task_type == "T2I" and ("1.3B" in base_model or "T2I" in base_model):
            filtered.append(lora["name"])
        elif not metadata.get("base_model"):
            filtered.append(lora["name"])  # Если не знаем, показываем
    
    return filtered

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def load_search_cache():
    """Загружает кэш поиска из файла"""
    if SEARCH_CACHE_FILE.exists():
        with open(SEARCH_CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_search_cache(cache):
    """Сохраняет кэш поиска в файл"""
    with open(SEARCH_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def get_installed_loras():
    """Возвращает список установленных LoRA с метаданными"""
    loras = []
    if LORA_DIR.exists():
        for item in LORA_DIR.iterdir():
            if item.is_dir():
                lora_files = list(item.glob("*.safetensors"))
                if lora_files:
                    loras.append({
                        "name": item.name,
                        "path": str(item),
                        "file": str(lora_files[0].name),
                        "size_mb": lora_files[0].stat().st_size / (1024 * 1024)
                    })
    return loras

def check_model_status():
    """Проверяет, какие модели уже скачаны"""
    status = {}
    for key, model in MODELS.items():
        model_dir = MODELS_DIR / key.replace("_", "-")
        if not model_dir.exists():
            model_dir = MODELS_DIR / key
        if model_dir.exists():
            has_files = len(list(model_dir.glob("*.safetensors"))) > 0
            status[key] = has_files
        else:
            status[key] = False
    return status

def download_model_with_hub(model_key, progress=gr.Progress()):
    """Скачивает модель через huggingface_hub с прогресс-баром"""
    model = MODELS[model_key]
    model_dir = MODELS_DIR / model_key
    
    try:
        progress(0, desc=f"Подготовка к скачиванию {model['name']}...")
        
        snapshot_download(
            repo_id=model["repo_id"],
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4
        )
        
        progress(1.0, desc="Готово!")
        return f"✅ Модель {model['name']} успешно скачана в {model_dir}"
    except Exception as e:
        return f"❌ Ошибка скачивания: {str(e)}"

def check_and_download_model_if_needed(model_key):
    """Проверяет наличие модели и скачивает при необходимости"""
    status = check_model_status()
    if not status.get(model_key, False):
        print(f"📦 Модель {model_key} не найдена, скачиваю...")
        return download_model_with_hub(model_key)
    return f"✅ Модель {model_key} уже существует"

# ============================================================
# ФУНКЦИИ ПОИСКА LORA НА HF/GITHUB
# ============================================================

def search_huggingface_lora(query, lora_type="all"):
    """Поиск LoRA на HuggingFace"""
    cache = load_search_cache()
    cache_key = f"hf_{query}_{lora_type}"
    
    if cache_key in cache and time.time() - cache[cache_key].get("timestamp", 0) < 300:
        return cache[cache_key]["results"]
    
    results = []
    search_queries = [f"Wan2.1 {query} lora", f"wan {query} lora"]
    
    for search_query in search_queries:
        try:
            models = api.list_models(
                search=search_query,
                sort="downloads",
                direction=-1,
                limit=20
            )
            for model in models:
                model_id_lower = model.model_id.lower()
                if "lora" in model_id_lower or "lora" in str(model.tags or []):
                    results.append({
                        "id": model.model_id,
                        "name": model.model_id.split("/")[-1],
                        "author": model.model_id.split("/")[0],
                        "downloads": getattr(model, "downloads", 0),
                        "likes": getattr(model, "likes", 0),
                        "tags": model.tags,
                        "url": f"https://huggingface.co/{model.model_id}"
                    })
        except Exception as e:
            print(f"Ошибка поиска HF: {e}")
    
    results.sort(key=lambda x: x.get("downloads", 0), reverse=True)
    
    cache[cache_key] = {"timestamp": time.time(), "results": results[:20]}
    save_search_cache(cache)
    
    return results[:20]

def search_github_lora(query):
    """Поиск репозиториев с LoRA на GitHub"""
    try:
        url = "https://api.github.com/search/repositories"
        params = {"q": f"Wan2.1 {query} lora", "sort": "stars", "order": "desc"}
        response = requests.get(url, params=params, headers={"Accept": "application/vnd.github.v3+json"})
        if response.status_code == 200:
            items = response.json().get("items", [])
            return [{
                "name": item["name"],
                "author": item["owner"]["login"],
                "stars": item["stargazers_count"],
                "url": item["html_url"],
                "description": item.get("description", "")
            } for item in items[:10]]
    except Exception as e:
        print(f"GitHub search error: {e}")
    return []

def download_lora_from_hf(lora_id, progress=gr.Progress()):
    """Скачивает LoRA из HuggingFace"""
    lora_path = LORA_DIR / lora_id.replace("/", "_")
    lora_path.mkdir(exist_ok=True, parents=True)
    
    try:
        progress(0, desc="Подготовка к скачиванию...")
        snapshot_download(
            repo_id=lora_id,
            local_dir=str(lora_path),
            allow_patterns=["*.safetensors", "*.bin", "*.json", "*.png", "*.jpg", "*.jpeg", "*.webp"]
        )
        progress(1.0, desc="Готово!")
        return f"✅ LoRA {lora_id} успешно скачан в {lora_path}"
    except Exception as e:
        return f"❌ Ошибка скачивания: {str(e)}"

# ============================================================
# ФУНКЦИИ ГЕНЕРАЦИИ
# ============================================================

def run_wan_generation(task, prompt, image, frames, steps, width, height, seed, lora_name, lora_strength, progress=gr.Progress()):
    """Запускает generate.py с прогресс-баром"""
    
    # Автоматическая проверка и скачивание модели
    if task == "Text-to-Video":
        model_check = check_and_download_model_if_needed("t2v_1.3b")
        if "Ошибка" in model_check:
            return model_check, None
    elif task == "Image-to-Video":
        model_check = check_and_download_model_if_needed("i2v_14b_fp8")
        if "Ошибка" in model_check:
            return model_check, None
    
    progress(0, desc="Подготовка...")
    
    # Базовые параметры
    cmd = [
        sys.executable, str(GENERATE_SCRIPT),
        "--offload_model", "True",
        "--t5_cpu", "True",
        "--frame_num", str(frames),
        "--sample_steps", str(steps),
        "--sample_shift", "8",
        "--sample_guide_scale", "6",
    ]
    
    progress(0.1, desc="Настройка параметров...")
    
    # LoRA (если выбрана)
    if lora_name and lora_name != "Без LoRA":
        lora_path = LORA_DIR / lora_name
        lora_files = list(lora_path.glob("*.safetensors"))
        if lora_files:
            rel_lora_path = f"models/lora/{lora_name}/{lora_files[0].name}"
            cmd.extend(["--lora_path", rel_lora_path])
            cmd.extend(["--lora_strength", str(lora_strength)])
            
            # Добавляем триггерные слова в промпт (если их нет)
            trigger_words = get_lora_trigger_words(lora_name)
            if trigger_words and trigger_words not in prompt:
                prompt = f"{trigger_words} {prompt}"
    
    # Выбор задачи
    if task == "Text-to-Video":
        cmd.extend(["--task", "t2v-1.3B"])
        cmd.extend(["--size", f"{width}*{height}"])
        cmd.extend(["--prompt", prompt])
        
    elif task == "Image-to-Video":
        cmd.extend(["--task", "i2v-14B"])
        cmd.extend(["--size", f"{width}*{height}"])
        cmd.extend(["--prompt", prompt])
        
        if image:
            temp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            shutil.copy(image, temp_img.name)
            cmd.extend(["--image", temp_img.name])
        else:
            return "❌ Ошибка: нужно загрузить изображение для I2V", None
            
    elif task == "Text-to-Image":
        cmd.extend(["--task", "t2v-1.3B"])
        cmd.extend(["--size", f"{width}*{height}"])
        cmd.extend(["--prompt", prompt])
        cmd.extend(["--frame_num", "1"])
    
    if seed and seed != -1:
        cmd.extend(["--seed", str(seed)])
    
    progress(0.2, desc="Запуск генерации...")
    
    # Запускаем генерацию
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            cwd=str(WAN_DIR)
        )
    except Exception as e:
        return f"❌ Ошибка запуска: {str(e)}", None
    
    # Мониторинг прогресса (симуляция)
    for i in range(20, 95):
        time.sleep(0.5)
        if process.poll() is not None:
            break
        progress(i / 100, desc=f"Генерация... {i}%")
    
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        return f"❌ Ошибка: {stderr}", None
    
    progress(0.95, desc="Поиск результата...")
    
    # Ищем результат
    output_files = sorted(OUTPUT_DIR.rglob("*.mp4"))
    if not output_files:
        output_files = sorted(OUTPUT_DIR.rglob("*.png"))
    
    progress(1.0, desc="Готово!")
    
    if output_files:
        return "✅ Готово!", str(output_files[-1])
    
    return "⚠️ Файл не найден", None

# ============================================================
# ФУНКЦИИ ДЛЯ UI
# ============================================================

def refresh_lora_dropdown(task_type="T2V"):
    """Обновляет список LoRA в выпадающем списке (с фильтрацией)"""
    loras = get_loras_by_task(task_type)
    return gr.Dropdown(choices=loras, value="Без LoRA")

def on_lora_select(lora_name, task_type):
    """Обработчик выбора LoRA: показывает превью и информацию"""
    if not lora_name or lora_name == "Без LoRA":
        return gr.Image(value=None, visible=False), "Выберите LoRA для просмотра информации", ""
    
    preview_path = get_lora_preview(lora_name)
    info_text = get_lora_info_text(lora_name)
    trigger_words = get_lora_trigger_words(lora_name)
    
    if preview_path:
        preview_update = gr.Image(value=preview_path, visible=True)
    else:
        preview_update = gr.Image(value=None, visible=False)
    
    return preview_update, info_text, trigger_words

def perform_search(query, lora_type):
    """Выполняет поиск LoRA и возвращает HTML"""
    if not query:
        return "Введите поисковый запрос", ""
    
    hf_results = search_huggingface_lora(query)
    gh_results = search_github_lora(query)
    
    html = '<div style="max-height: 500px; overflow-y: auto;">'
    
    if hf_results:
        html += '<h3>🤗 Hugging Face</h3>'
        for lora in hf_results[:10]:
            html += f'''
            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>📁 {lora["name"]}</strong><br>
                        <small>👤 {lora["author"]} | ⭐ {lora.get("likes", 0)} | ⬇️ {lora.get("downloads", 0)}</small><br>
                        <small>🔗 <a href="{lora["url"]}" target="_blank">{lora["id"]}</a></small>
                    </div>
                    <div>
                        <button onclick="alert(\'Скачайте вручную: huggingface-cli download {lora["id"]}\')" style="background: #4CAF50; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer;">💾 Скачать</button>
                    </div>
                </div>
            </div>
            '''
    
    if gh_results:
        html += '<h3>🐙 GitHub</h3>'
        for repo in gh_results[:5]:
            html += f'''
            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>📁 {repo["name"]}</strong><br>
                        <small>👤 {repo["author"]} | ⭐ {repo["stars"]}</small><br>
                        <small>{repo.get("description", "")[:100]}</small>
                    </div>
                    <div>
                        <a href="{repo["url"]}" target="_blank">
                            <button style="background: #333; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer;">🔗 Открыть</button>
                        </a>
                    </div>
                </div>
            </div>
            '''
    
    html += '</div>'
    
    if not hf_results and not gh_results:
        return "Ничего не найдено. Попробуйте другой запрос.", ""
    
    return f"Найдено: {len(hf_results)} на HuggingFace, {len(gh_results)} на GitHub", html

def update_installed_loras_table():
    """Обновляет таблицу установленных LoRA"""
    loras = get_installed_loras()
    if loras:
        data = []
        for lora in loras:
            trigger = get_lora_trigger_words(lora["name"]) or "-"
            data.append([lora["name"], trigger, f"{lora['size_mb']:.1f}"])
        return data
    return [["Нет установленных LoRA", "-", ""]]

def update_model_status():
    """Обновляет статус моделей"""
    return check_model_status()

# ============================================================
# СОЗДАНИЕ ИНТЕРФЕЙСА
# ============================================================

def create_interface():
    with gr.Blocks(title="Wan2.1 Ultimate UI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎬 Wan2.1 Ultimate UI")
        gr.Markdown("Лёгкая оболочка для RTX 3060 Ti 8GB | CPU + RAM компенсируют нехватку VRAM | Поддержка LoRA | Автоустановка | CivitAI интеграция")
        
        with gr.Tabs():
            # ========== ВКЛАДКА 1: ЗАГРУЗКА МОДЕЛЕЙ ==========
            with gr.TabItem("💾 Загрузка моделей"):
                gr.Markdown("### Скачайте модели перед использованием")
                gr.Markdown("Модели скачиваются автоматически при первой генерации, но вы можете скачать их заранее здесь.")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 📹 Text-to-Video (1.3B)")
                        gr.Markdown(f"Размер: ~{MODELS['t2v_1.3b']['size_gb']} GB | VRAM: {MODELS['t2v_1.3b']['required_vram']} GB")
                        gr.Markdown("✅ **РАБОТАЕТ на вашей RTX 3060 Ti 8GB**")
                        btn_download_t2v = gr.Button("⬇️ Скачать T2V 1.3B модель", variant="primary")
                        status_t2v = gr.Textbox(label="Статус", interactive=False)
                        
                    with gr.Column():
                        gr.Markdown("#### 🖼️ Image-to-Video (14B FP8)")
                        gr.Markdown(f"Размер: ~{MODELS['i2v_14b_fp8']['size_gb']} GB | VRAM: {MODELS['i2v_14b_fp8']['required_vram']} GB")
                        gr.Markdown("✅ **РАБОТАЕТ на вашей RTX 3060 Ti 8GB** (15-20 мин)")
                        btn_download_i2v = gr.Button("⬇️ Скачать I2V 14B FP8 модель", variant="primary")
                        status_i2v = gr.Textbox(label="Статус", interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("#### 🎬 Text-to-Video (14B)")
                        gr.Markdown(f"Размер: ~{MODELS['t2v_14b']['size_gb']} GB | VRAM: {MODELS['t2v_14b']['required_vram']} GB")
                        gr.Markdown("⚠️ **НЕ РАБОТАЕТ на 8GB** (требует 24GB VRAM)")
                        btn_download_t2v_14b = gr.Button("⬇️ Скачать T2V 14B модель", variant="secondary")
                        status_t2v_14b = gr.Textbox(label="Статус", interactive=False)
                
                gr.Markdown("### 📊 Статус моделей")
                refresh_status_btn = gr.Button("🔄 Обновить статус")
                model_status = gr.JSON(label="Скачанные модели")
                
                refresh_status_btn.click(fn=update_model_status, outputs=model_status)
                demo.load(fn=update_model_status, outputs=model_status)
                
                btn_download_t2v.click(fn=download_model_with_hub, inputs=gr.State("t2v_1.3b"), outputs=status_t2v, show_progress="full")
                btn_download_i2v.click(fn=download_model_with_hub, inputs=gr.State("i2v_14b_fp8"), outputs=status_i2v, show_progress="full")
                btn_download_t2v_14b.click(fn=download_model_with_hub, inputs=gr.State("t2v_14b"), outputs=status_t2v_14b, show_progress="full")
            
            # ========== ВКЛАДКА 2: ПОИСК LORA ==========
            with gr.TabItem("🔍 Поиск LoRA"):
                gr.Markdown("### 🔍 Поиск LoRA для Wan2.1 на Hugging Face и GitHub")
                gr.Markdown("Найдите LoRA для стилизации, ускорения или управления генерацией.")
                
                with gr.Row():
                    search_query = gr.Textbox(label="Поисковый запрос", placeholder="speed, anime, control, depth...", scale=3)
                    search_btn = gr.Button("🔍 Искать", variant="primary", scale=1)
                
                search_info = gr.Markdown("Введите запрос и нажмите 'Искать'")
                search_results = gr.HTML(label="Результаты поиска")
                
                gr.Markdown("---")
                gr.Markdown("### 📁 Установленные LoRA")
                installed_table = gr.Dataframe(headers=["Название", "Триггерные слова", "Размер (MB)"])
                refresh_lora_table_btn = gr.Button("🔄 Обновить список")
                
                search_btn.click(fn=perform_search, inputs=[search_query, gr.State("all")], outputs=[search_info, search_results])
                refresh_lora_table_btn.click(fn=update_installed_loras_table, outputs=installed_table)
                demo.load(fn=update_installed_loras_table, outputs=installed_table)
            
            # ========== ВКЛАДКА 3: TEXT-TO-VIDEO ==========
            with gr.TabItem("📝 Text-to-Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_t2v = gr.Textbox(label="📝 Промпт", lines=3, placeholder="Опишите видео... Например: 'Космонавт едет на лошади по Марсу'")
                        
                        gr.Markdown("### 🎯 LoRA")
                        lora_select_t2v = gr.Dropdown(label="Выберите LoRA", choices=["Без LoRA"], value="Без LoRA")
                        lora_strength_t2v = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Сила LoRA (0.5-1.0 рекомендуется)")
                        
                        with gr.Row():
                            refresh_lora_t2v = gr.Button("🔄 Обновить список LoRA", size="sm")
                            add_triggers_t2v = gr.Button("✨ Добавить триггерные слова в промпт", size="sm")
                        
                        # Превью LoRA
                        lora_preview_t2v = gr.Image(label="🖼️ Превью LoRA (локальное или с CivitAI)", visible=False)
                        lora_info_t2v = gr.Markdown("Выберите LoRA для просмотра информации")
                        lora_trigger_display_t2v = gr.Textbox(label="🔑 Триггерные слова", visible=False, interactive=False)
                        
                        with gr.Row():
                            width_t2v = gr.Slider(384, 832, value=832, step=64, label="Ширина")
                            height_t2v = gr.Slider(384, 832, value=480, step=64, label="Высота")
                        with gr.Row():
                            frames_t2v = gr.Slider(17, 81, value=33, step=4, label="Кадры (17-81)")
                            steps_t2v = gr.Slider(10, 50, value=25, step=1, label="Шаги сэмплинга")
                        seed_t2v = gr.Number(value=-1, label="Seed (-1 = случайный)", precision=0)
                        
                        btn_t2v = gr.Button("🚀 Сгенерировать видео", variant="primary")
                        progress_t2v = gr.Progress(show_label=True)
                        
                    with gr.Column(scale=1):
                        output_t2v = gr.Video(label="🎬 Результат")
                        status_t2v = gr.Textbox(label="📊 Статус", interactive=False)
                
                # Обработчики
                def refresh_t2v_loras():
                    return refresh_lora_dropdown("T2V")
                
                def on_t2v_lora_select(lora_name):
                    return on_lora_select(lora_name, "T2V")
                
                refresh_lora_t2v.click(fn=refresh_t2v_loras, outputs=lora_select_t2v)
                lora_select_t2v.change(
                    fn=on_t2v_lora_select,
                    inputs=lora_select_t2v,
                    outputs=[lora_preview_t2v, lora_info_t2v, lora_trigger_display_t2v]
                )
                add_triggers_t2v.click(
                    fn=lambda lora: get_lora_trigger_words(lora),
                    inputs=lora_select_t2v,
                    outputs=prompt_t2v
                )
                
                btn_t2v.click(
                    fn=run_wan_generation,
                    inputs=[gr.State("Text-to-Video"), prompt_t2v, gr.State(None), frames_t2v, steps_t2v, width_t2v, height_t2v, seed_t2v, lora_select_t2v, lora_strength_t2v],
                    outputs=[status_t2v, output_t2v],
                    show_progress="full"
                )
            
            # ========== ВКЛАДКА 4: IMAGE-TO-VIDEO ==========
            with gr.TabItem("🖼️ Image-to-Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_i2v = gr.Image(label="🖼️ Загрузите изображение", type="filepath")
                        prompt_i2v = gr.Textbox(label="📝 Промпт (описание движения)", lines=3, placeholder="Что происходит на видео?")
                        
                        gr.Markdown("### 🎯 LoRA")
                        lora_select_i2v = gr.Dropdown(label="Выберите LoRA", choices=["Без LoRA"], value="Без LoRA")
                        lora_strength_i2v = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Сила LoRA")
                        
                        with gr.Row():
                            refresh_lora_i2v = gr.Button("🔄 Обновить список LoRA", size="sm")
                            add_triggers_i2v = gr.Button("✨ Добавить триггерные слова в промпт", size="sm")
                        
                        lora_preview_i2v = gr.Image(label="🖼️ Превью LoRA (локальное или с CivitAI)", visible=False)
                        lora_info_i2v = gr.Markdown("Выберите LoRA для просмотра информации")
                        lora_trigger_display_i2v = gr.Textbox(label="🔑 Триггерные слова", visible=False, interactive=False)
                        
                        with gr.Row():
                            width_i2v = gr.Slider(384, 832, value=832, step=64, label="Ширина")
                            height_i2v = gr.Slider(384, 832, value=480, step=64, label="Высота")
                        with gr.Row():
                            frames_i2v = gr.Slider(17, 81, value=33, step=4, label="Кадры")
                            steps_i2v = gr.Slider(10, 50, value=25, step=1, label="Шаги")
                        seed_i2v = gr.Number(value=-1, label="Seed (-1 = случайный)", precision=0)
                        
                        btn_i2v = gr.Button("🚀 Сгенерировать видео", variant="primary")
                        progress_i2v = gr.Progress(show_label=True)
                        
                    with gr.Column(scale=1):
                        output_i2v = gr.Video(label="🎬 Результат")
                        status_i2v = gr.Textbox(label="📊 Статус", interactive=False)
                
                def refresh_i2v_loras():
                    return refresh_lora_dropdown("I2V")
                
                def on_i2v_lora_select(lora_name):
                    return on_lora_select(lora_name, "I2V")
                
                refresh_lora_i2v.click(fn=refresh_i2v_loras, outputs=lora_select_i2v)
                lora_select_i2v.change(
                    fn=on_i2v_lora_select,
                    inputs=lora_select_i2v,
                    outputs=[lora_preview_i2v, lora_info_i2v, lora_trigger_display_i2v]
                )
                add_triggers_i2v.click(
                    fn=lambda lora: get_lora_trigger_words(lora),
                    inputs=lora_select_i2v,
                    outputs=prompt_i2v
                )
                
                btn_i2v.click(
                    fn=run_wan_generation,
                    inputs=[gr.State("Image-to-Video"), prompt_i2v, image_i2v, frames_i2v, steps_i2v, width_i2v, height_i2v, seed_i2v, lora_select_i2v, lora_strength_i2v],
                    outputs=[status_i2v, output_i2v],
                    show_progress="full"
                )
            
            # ========== ВКЛАДКА 5: TEXT-TO-IMAGE ==========
            with gr.TabItem("🎨 Text-to-Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_t2i = gr.Textbox(label="📝 Промпт", lines=3, placeholder="Опишите изображение...")
                        
                        gr.Markdown("### 🎯 LoRA")
                        lora_select_t2i = gr.Dropdown(label="Выберите LoRA", choices=["Без LoRA"], value="Без LoRA")
                        lora_strength_t2i = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Сила LoRA")
                        
                        with gr.Row():
                            refresh_lora_t2i = gr.Button("🔄 Обновить список LoRA", size="sm")
                            add_triggers_t2i = gr.Button("✨ Добавить триггерные слова в промпт", size="sm")
                        
                        lora_preview_t2i = gr.Image(label="🖼️ Превью LoRA (локальное или с CivitAI)", visible=False)
                        lora_info_t2i = gr.Markdown("Выберите LoRA для просмотра информации")
                        
                        with gr.Row():
                            width_t2i = gr.Slider(256, 1024, value=832, step=64, label="Ширина")
                            height_t2i = gr.Slider(256, 1024, value=832, step=64, label="Высота")
                        steps_t2i = gr.Slider(10, 50, value=25, step=1, label="Шаги")
                        seed_t2i = gr.Number(value=-1, label="Seed (-1 = случайный)", precision=0)
                        
                        btn_t2i = gr.Button("🚀 Сгенерировать изображение", variant="primary")
                        progress_t2i = gr.Progress(show_label=True)
                        
                    with gr.Column(scale=1):
                        output_t2i = gr.Image(label="🎨 Результат")
                        status_t2i = gr.Textbox(label="📊 Статус", interactive=False)
                
                def refresh_t2i_loras():
                    return refresh_lora_dropdown("T2I")
                
                def on_t2i_lora_select(lora_name):
                    return on_lora_select(lora_name, "T2I")
                
                refresh_lora_t2i.click(fn=refresh_t2i_loras, outputs=lora_select_t2i)
                lora_select_t2i.change(
                    fn=on_t2i_lora_select,
                    inputs=lora_select_t2i,
                    outputs=[lora_preview_t2i, lora_info_t2i, gr.State()]
                )
                add_triggers_t2i.click(
                    fn=lambda lora: get_lora_trigger_words(lora),
                    inputs=lora_select_t2i,
                    outputs=prompt_t2i
                )
                
                btn_t2i.click(
                    fn=run_wan_generation,
                    inputs=[gr.State("Text-to-Image"), prompt_t2i, gr.State(None), gr.State(1), steps_t2i, width_t2i, height_t2i, seed_t2i, lora_select_t2i, lora_strength_t2i],
                    outputs=[status_t2i, output_t2i],
                    show_progress="full"
                )
    
    return demo

# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Wan2.1 Ultimate UI")
    parser.add_argument("--port", type=int, default=None, help="Порт для запуска (по умолчанию автоматический поиск)")
    args = parser.parse_args()
    
    ui = create_interface()
    
    # Определяем порт
    if args.port:
        port = args.port
        print(f"🔧 Использую указанный порт: {port}")
    else:
        port = find_free_port(start_port=7860, max_attempts=10)
        if port is None:
            print("❌ Не удалось найти свободный порт. Попробуйте закрыть некоторые программы.")
            sys.exit(1)
        print(f"✅ Найден свободный порт: {port}")
    
    print(f"🌐 Интерфейс будет доступен по адресу: http://127.0.0.1:{port}")
    
    # Автозапуск браузера
    def open_browser_with_port():
        time.sleep(2)
        webbrowser.open(f"http://127.0.0.1:{port}")
    
    threading.Thread(target=open_browser_with_port, daemon=True).start()
    
    # Запуск сервера
    ui.launch(share=False, server_name="127.0.0.1", server_port=port)
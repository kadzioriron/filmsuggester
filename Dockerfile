# Берем образ питона
FROM python:3.9

# Создаем рабочую папку
WORKDIR /app

# Копируем requirements и ставим либы
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все остальные файлы проекта
COPY . .

# Даем права (требование Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Команда для запуска сервера (Хаггинг Фейс слушает порт 7860)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
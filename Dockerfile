FROM python:3.12-slim

# Настройка переменных окружения Python и Streamlit
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Установка системных утилит (для сборки бинарных пакетов)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование требований и обновление базового тулинга сборки
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Предзагрузка словарей NLTK
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

# Копирование исходного кода приложения
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
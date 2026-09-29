FROM python:3.12-slim

WORKDIR /app

# Системные зависимости для сборки C-расширений
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Добавляем punkt_tab в список загружаемых ресурсов NLTK
RUN python -m nltk.downloader punkt punkt_tab stopwords

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

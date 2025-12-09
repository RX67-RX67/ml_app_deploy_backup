# streamlit_app/Dockerfile – Streamlit frontend
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (same requirements.txt as backend)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entire project (so Streamlit can import src, etc.)
COPY . .

# Streamlit runs on 8501
EXPOSE 8501

# In docker-compose we set API_URL=http://snapstyle-backend:8000
ENV API_URL=http://snapstyle-backend:8000/snapstyle
ENV AUTH_URL=http://snapstyle-backend:8000/api/auth

# if your main app file is at project root as app.py
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# (change app.py → whatever your actual UI file is)
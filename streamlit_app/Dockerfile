# Streamlit frontend
FROM python:3.11-slim

WORKDIR /app

# System-level deps
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install ONLY frontend dependencies
COPY streamlit_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ONLY the Streamlit app folder 
COPY streamlit_app /app

# API endpoint environment variable
ENV API_URL=http://snapstyle-backend:8000

EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

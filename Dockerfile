# -------------------------------------------------------
# Base Image
# -------------------------------------------------------
FROM python:3.10-slim

# -------------------------------------------------------
# System Dependencies
# -------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------
# Working directory
# -------------------------------------------------------
WORKDIR /app

# -------------------------------------------------------
# Copy project files
# -------------------------------------------------------
COPY . /app

# -------------------------------------------------------
# Install Python dependencies
# -------------------------------------------------------
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# -------------------------------------------------------
# Streamlit Settings
# -------------------------------------------------------
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# -------------------------------------------------------
# Expose port
# -------------------------------------------------------
EXPOSE 7860

# -------------------------------------------------------
# Run Streamlit App
# -------------------------------------------------------
CMD ["streamlit", "run", "streamlit_app/app_new.py", "--server.port=7860", "--server.address=0.0.0.0"]

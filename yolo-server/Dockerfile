# Dockerfile (optimized for FastAPI+YOLOv8 on Cloud Run)
FROM python:3.10-slim-buster

# Install OS libraries for OpenCV/GL support (avoids libGL errors)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 libfontconfig1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory and copy requirements
WORKDIR /app
COPY requirements.txt ./

# Install Python dependencies (includes ultralytics, fastapi, uvicorn, python-multipart, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . /app

# (Optional) Expose the port (Cloud Run uses 8080 by default)
EXPOSE 8080

# Correct CMD: point to server.py's app, not main.py
# Use uvicorn (or gunicorn) to run the FastAPI app on port 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]


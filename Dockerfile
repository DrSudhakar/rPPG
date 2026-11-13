# SAMA research demonstrator (web app) and reproducible analysis environment.
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git make && rm -rf /var/lib/apt/lists/*
WORKDIR /sama
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY . .
EXPOSE 8000
# Research use only. Not a medical device.
CMD ["python", "-m", "app.server", "--host", "0.0.0.0", "--port", "8000"]

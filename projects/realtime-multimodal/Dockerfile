FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/home/app GRADIO_ANALYTICS_ENABLED=False
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends espeak ffmpeg && rm -rf /var/lib/apt/lists/*
COPY projects/realtime-multimodal/requirements*.txt /tmp/deps/
RUN pip install --no-cache-dir -r /tmp/deps/requirements-voice.txt
RUN useradd --create-home --uid 1000 app
COPY --chown=app:app . .
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/')"
CMD ["python", "projects/realtime-multimodal/app.py"]

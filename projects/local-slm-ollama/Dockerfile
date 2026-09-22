FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/home/app
WORKDIR /app
COPY requirements-apps.txt ./
RUN pip install --no-cache-dir -r requirements-apps.txt
RUN useradd --create-home --uid 1000 app
COPY --chown=app:app . .
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"
CMD ["streamlit", "run", "projects/local-slm-ollama/app.py", "--server.address=0.0.0.0", "--server.port=8501"]

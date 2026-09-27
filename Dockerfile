FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && useradd --create-home --uid 10001 app
COPY common ./common
COPY backend ./backend
COPY generator ./generator
COPY ml_service ./ml_service
COPY training ./training
COPY tools/load_context.py tools/export_learning_data.py ./tools/
RUN mkdir -p /var/lib/mos-transport && chown -R app:app /var/lib/mos-transport
USER app
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]

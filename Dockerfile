# Fraud Detection API — containerized scoring service
#
# Build:  docker build -t fraud-detection-api .
# Run:    docker run -p 8000:8000 fraud-detection-api
# Test:   curl http://localhost:8000/health

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer) so Docker can cache this step
# and only re-run it when requirements actually change, not on every code edit.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Now copy the application code and the trained model artifacts
COPY src/ src/
COPY models/ models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

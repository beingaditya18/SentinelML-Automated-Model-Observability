FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for compiling numerical libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and reference datasets
COPY api/ api/
COPY model/ model/
COPY data/ data/
COPY monitoring/ monitoring/
COPY prometheus/ prometheus/

# Expose API and Exporter ports
EXPOSE 5000
EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "5000"]

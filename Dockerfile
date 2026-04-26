FROM python:3.10-slim

WORKDIR /app

# System deps for matplotlib and XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY . .

# Generate data & train model if no versioned artifact dirs exist
RUN artifact_dirs=$(find artifacts -mindepth 1 -maxdepth 1 -type d 2>/dev/null); \
    if [ -z "$artifact_dirs" ]; then \
        python scripts/generate_sample_data.py && \
        python main.py --input data/customers.csv; \
    fi

ENV MPLBACKEND=Agg
EXPOSE 5000

CMD ["gunicorn", "webapp:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120", "--max-requests", "50", "--max-requests-jitter", "10"]

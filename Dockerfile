FROM python:3.10-slim

WORKDIR /app

# System deps for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY . .

# Generate data & train model if no artifacts shipped
RUN if [ ! -d "artifacts" ] || [ -z "$(ls -A artifacts 2>/dev/null)" ]; then \
        python scripts/generate_sample_data.py && \
        python main.py --input data/customers.csv; \
    fi

ENV MPLBACKEND=Agg
EXPOSE 5000

CMD ["gunicorn", "webapp:app", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120"]

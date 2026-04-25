#!/usr/bin/env bash
# Render build script — runs during each deploy
set -o errexit

pip install --upgrade pip
pip install -r requirements-prod.txt

# Generate sample data and train model if artifacts don't exist
if [ ! -d "artifacts" ] || [ -z "$(ls -A artifacts 2>/dev/null)" ]; then
    echo "No artifacts found — generating data and training model..."
    python scripts/generate_sample_data.py
    python main.py --input data/customers.csv
    echo "Training complete."
else
    echo "Artifacts found — skipping training."
fi

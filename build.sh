#!/usr/bin/env bash
# Render build script — runs during each deploy
set -o errexit

pip install --upgrade pip
pip install -r requirements-prod.txt

# Generate sample data and train model if no versioned artifact dirs exist
# (ignore .gitkeep and other dotfiles when checking)
artifact_dirs=$(find artifacts -mindepth 1 -maxdepth 1 -type d 2>/dev/null)
if [ -z "$artifact_dirs" ]; then
    echo "No artifacts found — generating data and training model..."
    python scripts/generate_sample_data.py
    python main.py --input data/customers.csv
    echo "Training complete."
else
    echo "Artifacts found — skipping training."
fi

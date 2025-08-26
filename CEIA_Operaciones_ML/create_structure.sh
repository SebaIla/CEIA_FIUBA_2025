#!/bin/bash

# Create the main project directory
mkdir -p collision-prediction-project

# Navigate into the project directory
cd collision-prediction-project

# Create the directory structure
mkdir -p data/raw
mkdir -p data/processed
mkdir -p notebooks
mkdir -p src
mkdir -p models
mkdir -p mlruns

# Create empty files
touch environment.yml
touch metaflow_flow.py
touch README.md
touch notebooks/collision_model.ipynb
touch src/__init__.py
touch src/data_ingestion.py
touch src/data_processing.py
touch src/train.py

# Output the created structure
echo "Folder structure created:"
find . -type d -print | sed -e 's;[^/]*/;|____;g;s;____|; |;g'
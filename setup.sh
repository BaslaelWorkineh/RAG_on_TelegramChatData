#!/bin/bash

# Create and activate virtual environment
python3 -m venv rag_env
source rag_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Download SpaCy language model
python -m spacy download en_core_web_sm

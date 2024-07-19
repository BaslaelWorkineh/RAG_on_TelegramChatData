#!/bin/bash

python3 -m venv rag_env
source rag_env/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

python -m spacy download en_core_web_sm

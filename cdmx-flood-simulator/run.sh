#!/usr/bin/env bash
set -e
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
streamlit run app.py --server.port=8505

@echo off
python -m venv .venv
call .venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
streamlit run app.py --server.port=8501

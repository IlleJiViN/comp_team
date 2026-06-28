@echo off
echo Starting AI Search Server V10...
call .venv\Scripts\activate
uvicorn ai_search_v10:app --host 0.0.0.0 --port 8000 --reload

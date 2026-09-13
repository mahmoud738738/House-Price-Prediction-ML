@echo off
title House Price Backend (FastAPI)
echo Starting FastAPI Backend on http://localhost:8000 ...
cd backend
python -m uvicorn app.main:app --reload --port 8000
pause

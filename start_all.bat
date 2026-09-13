@echo off
title House Price Predictor Launcher
echo ========================================================
echo Starting Indian House Price Prediction Web App...
echo ========================================================

start "House Price Backend (FastAPI)" cmd /k "cd backend && python -m uvicorn app.main:app --reload --port 8000"
timeout /t 3 /nobreak >nul

start "House Price Frontend (React + Vite)" cmd /k "cd frontend && npm run dev"
timeout /t 3 /nobreak >nul

echo Opening browser at http://localhost:5173 ...
start http://localhost:5173

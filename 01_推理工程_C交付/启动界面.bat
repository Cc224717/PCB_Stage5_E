@echo off
cd /d "%~dp0"
echo Starting PCB UI at http://127.0.0.1:8765
".\.venv\Scripts\python.exe" -m streamlit run ui_app.py --server.port 8765 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
pause

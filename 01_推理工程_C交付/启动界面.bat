@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "VENV=D:\PCB_envs\pcb_stage5_cu124"
set "PY=%VENV%\Scripts\python.exe"
set "VIRTUAL_ENV=%VENV%"
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH=%VENV%\Lib\site-packages"
set "PATH=%VENV%\Scripts;%PATH%"

if not exist "%PY%" (
  echo [ERROR] 找不到 GPU 环境: %PY%
  pause
  exit /b 1
)

echo ========================================
echo  PCB UI 启动（强制 GPU 环境）
echo  Python: %PY%
echo  URL:    http://127.0.0.1:8765
echo ========================================

"%PY%" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 2)"
if errorlevel 2 (
  echo [WARN] 当前环境 CUDA 不可用，界面仍会启动但会走 CPU
) else if errorlevel 1 (
  echo [ERROR] PyTorch 检查失败，请确认 D:\PCB_envs\pcb_stage5_cu124
  pause
  exit /b 1
)

REM 先清掉旧占用，避免连到错误的 python 进程
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8765" ^| findstr LISTENING') do (
  echo Killing old PID %%p on 8765
  taskkill /F /PID %%p >nul 2>&1
)

"%PY%" -m streamlit run ui_app.py --server.port 8765 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
pause

@echo off
title Neon Grid Space Simulation - MuJoCo
echo ==========================================================
echo Starting Neon Grid Space Simulation - Environment Creation
echo ==========================================================
echo.

set PYTHON_EXE=C:\Users\Dell\AppData\Local\Programs\Python\Python312\python.exe

:: Verify Python installation
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python 3.12 executable was not found at:
    echo %PYTHON_EXE%
    echo Checking if default 'python' command is available...
    where python >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set PYTHON_EXE=python
        echo [INFO] Using system default 'python' instead.
    ) else (
        echo [ERROR] Python is not found on your system PATH or at the specified location.
        echo Please ensure Python is installed and configured.
        pause
        exit /b 1
    )
)

echo [INFO] Using Python: %PYTHON_EXE%
echo.

:: Verify/install packages
echo [INFO] Checking dependencies (mujoco, pillow)...
"%PYTHON_EXE%" -c "import mujoco; print('  - mujoco version:', mujoco.__version__)" 2>nul
if %ERRORLEVEL% neq 0 (
    echo [INFO] Installing missing 'mujoco' package...
    "%PYTHON_EXE%" -m pip install mujoco
)

"%PYTHON_EXE%" -c "import PIL; print('  - pillow version:', PIL.__version__)" 2>nul
if %ERRORLEVEL% neq 0 (
    echo [INFO] Installing missing 'pillow' package...
    "%PYTHON_EXE%" -m pip install pillow
)
echo [INFO] Dependencies checked.
echo.

:: Generate visual assets
echo [INFO] Running asset generator (generate_assets.py)...
"%PYTHON_EXE%" generate_assets.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Visual asset generation failed.
    pause
    exit /b %ERRORLEVEL%
)
echo.

echo [INFO] Populating dynamic stone formations (add_rocks.py)...
"%PYTHON_EXE%" add_rocks.py
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Dynamic stone generation failed, using existing scene.xml structure.
)
echo.

:: Run simulation
echo [INFO] Launching MuJoCo Simulator (simulate.py)...
"%PYTHON_EXE%" simulate.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Simulator exited with an error.
)

echo.
echo ==========================================================
echo Simulation ended. Press any key to close this window.
echo ==========================================================
pause

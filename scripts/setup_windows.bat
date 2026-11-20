@echo off
REM ============================================================================
REM  Plant Disease Classification - Windows setup (no administrator rights needed)
REM  Creates .venv, installs pinned dependencies and verifies TensorFlow imports.
REM ============================================================================
setlocal enableextensions
cd /d "%~dp0.."
echo.
echo === Plant Disease Classification: setup ===
echo Project folder: %CD%
echo.

set "PYCMD="
py -3.12 --version >nul 2>&1
if not errorlevel 1 set "PYCMD=py -3.12"
if not defined PYCMD (
    python --version >nul 2>&1
    if not errorlevel 1 set "PYCMD=python"
)
if not defined PYCMD goto :nopython

%PYCMD% -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)"
if errorlevel 1 goto :badversion
echo Python check passed:
%PYCMD% --version

if exist ".venv\Scripts\activate.bat" (
    echo Virtual environment .venv already exists - reusing it.
) else (
    echo Creating virtual environment .venv ...
    %PYCMD% -m venv .venv
    if errorlevel 1 goto :fail
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :fail

echo Upgrading pip ...
python -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo Installing dependencies (TensorFlow is large; this can take several minutes) ...
python -m pip install -r requirements-dev.txt
if errorlevel 1 goto :fail

echo Verifying installation ...
python -c "import tensorflow as tf, keras, flask, numpy, PIL, sklearn, pandas, matplotlib; print('TensorFlow', tf.__version__, '| Keras', keras.__version__, '| Flask', flask.__version__)"
if errorlevel 1 goto :fail

echo.
echo SETUP COMPLETE.
echo Next steps:
echo   1. scripts\train_windows.bat      (downloads PlantVillage and trains; or use the Colab notebook)
echo   2. scripts\run_windows.bat        (starts the web app at http://127.0.0.1:5000)
echo.
pause
exit /b 0

:nopython
echo ERROR: Python was not found.
echo Install Python 3.12 (64-bit) from https://www.python.org/downloads/windows/
echo Choose "Install for me only" (no administrator rights) and tick "Add python.exe to PATH".
goto :fail

:badversion
echo ERROR: Python 3.12 is required (the pinned numpy 2.5 needs 3.12 or newer and this project was tested on 3.12).
echo Your version is:
%PYCMD% --version
echo Install Python 3.12 from https://www.python.org/downloads/windows/ and run this script again.
goto :fail

:fail
echo.
echo SETUP FAILED. Read the message above, fix the problem and run this script again.
pause
exit /b 1

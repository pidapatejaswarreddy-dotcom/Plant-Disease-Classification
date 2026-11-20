@echo off
REM ============================================================================
REM  Start the Flask web application. Open http://127.0.0.1:5000 in your browser.
REM ============================================================================
setlocal enableextensions
cd /d "%~dp0.."
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: virtual environment not found. Run scripts\setup_windows.bat first.
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"

if not exist "models\plant_disease_model.keras" (
    echo NOTE: no trained model found in the models folder.
    echo The app will start but show setup instructions instead of predictions.
    echo Train with scripts\train_windows.bat or copy the files from Colab into models\ .
    echo.
)
echo Starting the web app ...  (press Ctrl+C to stop)
echo Browser address: http://127.0.0.1:5000
python app.py
if errorlevel 1 (
    echo.
    echo The application stopped with an error. See the messages above.
)
pause

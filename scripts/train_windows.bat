@echo off
REM ============================================================================
REM  Train the model locally (CPU). Downloads + verifies PlantVillage if needed.
REM  Usage:  scripts\train_windows.bat                      (10 epochs + 5 fine-tune epochs)
REM          scripts\train_windows.bat --epochs 5 --fine-tune-epochs 0
REM  Training on CPU can take many hours; Google Colab (GPU) is much faster.
REM ============================================================================
setlocal enableextensions
cd /d "%~dp0.."
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: virtual environment not found. Run scripts\setup_windows.bat first.
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"

if not exist "dataset\dataset_info.json" (
    echo Dataset not prepared yet - downloading and verifying PlantVillage ...
    python training\prepare_dataset.py --source auto
    if errorlevel 1 (
        echo.
        echo Automatic download failed. Download the dataset manually and run ONE of:
        echo   python training\prepare_dataset.py --source zip    --zip    "C:\path\to\dataset.zip"
        echo   python training\prepare_dataset.py --source folder --folder "C:\path\to\folder"
        echo Then run this script again.
        pause
        exit /b 1
    )
)

set "TRAIN_ARGS=%*"
if "%TRAIN_ARGS%"=="" set "TRAIN_ARGS=--epochs 10 --fine-tune-epochs 5"
echo Starting training with: %TRAIN_ARGS%
python training\train.py %TRAIN_ARGS%
if errorlevel 1 (
    echo.
    echo TRAINING FAILED OR WAS NOT COMPLETED. No model was installed. See the messages above.
    pause
    exit /b 1
)
echo.
echo Training finished. Model files are in the models folder. Start the app with scripts\run_windows.bat
pause
exit /b 0

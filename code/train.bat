@echo off
echo ================================================================================
echo ConLingo 2.0 - Model Training
echo ================================================================================
echo.

if not exist venv (
    echo ERROR: Virtual environment not found
    echo Please run setup_windows.bat first
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting training...
echo This will take approximately 1-2 hours on a GPU
echo.

python scripts\train_model.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Training failed
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Training Complete!
echo ================================================================================
echo.
echo Next step: Run test.bat to test the model
echo.
pause

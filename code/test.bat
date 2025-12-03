@echo off
echo ================================================================================
echo ConLingo 2.0 - Model Testing
echo ================================================================================
echo.

if not exist venv (
    echo ERROR: Virtual environment not found
    echo Please run setup_windows.bat first
    pause
    exit /b 1
)

if not exist trained_model\final_model (
    echo ERROR: Trained model not found
    echo Please run train.bat first
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Running tests...
echo.

python scripts\test_model.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Testing failed
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Testing Complete!
echo ================================================================================
echo.
echo Results saved to: test_results.txt
echo Opening results file...
echo.

if exist test_results.txt (
    notepad test_results.txt
) else (
    echo ERROR: Results file not found
)

pause

@echo off
echo ================================================================================
echo ConLingo 2.0 - Windows Setup Script
echo ================================================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or higher from python.org
    pause
    exit /b 1
)

echo Step 1: Checking Python Version
python -c "import sys; v=sys.version_info; exit(0 if v.major==3 and 10<=v.minor<=12 else 1)"
if %errorlevel% neq 0 (
    echo ERROR: Python 3.10, 3.11, or 3.12 is required
    echo Your version:
    python --version
    echo.
    echo PyTorch does not yet support Python 3.13+
    echo Please install Python 3.11 from: https://www.python.org/downloads/release/python-3119/
    pause
    exit /b 1
)
python --version
echo Python version OK

echo.
echo ================================================================================
echo Step 2: Creating Virtual Environment
echo ================================================================================
if exist venv (
    echo Virtual environment already exists, skipping creation
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully
)

echo.
echo ================================================================================
echo Step 3: Activating Virtual Environment
echo ================================================================================
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated

echo.
echo ================================================================================
echo Step 4: Upgrading pip
echo ================================================================================
python -m pip install --upgrade pip
echo pip upgraded successfully

echo.
echo ================================================================================
echo Step 5: Installing Dependencies
echo ================================================================================
echo This may take 5-10 minutes...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully

echo.
echo ================================================================================
echo Step 6: Setting up HuggingFace Authentication
echo ================================================================================
echo.
echo You need a HuggingFace token to download the LLaMA model.
echo.
echo To get a token:
echo 1. Go to: https://huggingface.co/settings/tokens
echo 2. Create a new token with READ access
echo 3. Accept the LLaMA 3 license at: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
echo.
set /p HF_TOKEN="Enter your HuggingFace token: "

if "%HF_TOKEN%"=="" (
    echo ERROR: Token cannot be empty
    pause
    exit /b 1
)

python -c "from huggingface_hub import login; login(token='%HF_TOKEN%')"
if %errorlevel% neq 0 (
    echo ERROR: Failed to authenticate with HuggingFace
    pause
    exit /b 1
)

setx HF_TOKEN "%HF_TOKEN%" >nul
echo HuggingFace authentication successful

echo.
echo ================================================================================
echo Setup Complete!
echo ================================================================================
echo.
echo Next steps:
echo 1. Run train.bat to train the model (takes 1-2 hours on GPU)
echo 2. Run test.bat to test the model (takes 2-3 minutes)
echo.
pause

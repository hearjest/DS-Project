@echo off
REM Windows batch script to run the project pipeline
REM Alternative to Makefile for Windows users

echo ========================================
echo Stock Price Prediction Project
echo ========================================

echo.
echo Step 1: Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)

echo.
echo Step 2: Downloading stock data...
python getData.py
if errorlevel 1 (
    echo ERROR: Failed to download data
    exit /b 1
)

echo.
echo Step 3: Extracting features...
python feature_extraction_enhanced.py
if errorlevel 1 (
    echo ERROR: Failed to extract features
    exit /b 1
)

echo.
echo Step 4: Executing Jupyter notebook...
python -m jupyter nbconvert --to notebook --execute --inplace s3.4testsentiment2.ipynb
if errorlevel 1 (
    echo ERROR: Failed to execute notebook
    exit /b 1
)

echo.
echo ========================================
echo Pipeline completed successfully!
echo ========================================
echo.
echo All steps completed including notebook execution.
echo.

pause

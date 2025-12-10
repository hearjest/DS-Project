@echo off
REM Windows batch script to run the project pipeline
<<<<<<< HEAD
REM Alternative to Makefile for Windows users 
=======
REM Alternative to Makefile for Windows users
>>>>>>> 5da5bcf (update)

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
echo ========================================
echo Pipeline completed successfully!
echo ========================================
echo.
echo Next steps:
echo   1. Open s3.4testsentiment2.ipynb in Jupyter
echo   2. Run all cells to train models and generate results
echo.

pause


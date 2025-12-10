@echo off
REM Windows batch script to run tests
REM Alternative to 'make test' for Windows users

echo ========================================
echo Running Test Suite
echo ========================================

python test_project.py

if errorlevel 1 (
    echo.
    echo Tests completed with errors
    exit /b 1
) else (
    echo.
    echo All tests passed!
    exit /b 0
)

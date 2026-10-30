@echo off
echo ======================================================================
echo    AUTONOMOUS SMART TRAFFIC MANAGEMENT - GITHUB UPLOADER
echo    Target: https://github.com/Bhargav-garg/IMPRENDITORE-.git
echo ======================================================================
echo.
git push -u origin main
echo.
if %errorlevel% neq 0 (
    echo.
    echo If prompted, enter your GitHub Personal Access Token or sign in via browser.
) else (
    echo [SUCCESS] Code uploaded successfully to GitHub!
)
pause

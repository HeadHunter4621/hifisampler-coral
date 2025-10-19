@echo off
setlocal

rem --- Set up and verify the UV environment ---
call :SetupUv
if %errorlevel% neq 0 (
    echo.
    echo Fatal error during uv setup. The script cannot continue.
    pause
    goto :eof
)

rem --- Start the service ---
uv run launch_server.py

endlocal
goto :eof


:SetupUv
    echo --- Checking for uv...

    rem Attempt to execute `uv` directly. If successful, it indicates everything is ready and you can return immediately.
    uv --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo uv is already installed and accessible.
        exit /b 0
    )

    rem If the command fails, the installation process will begin.
    echo uv not found or not working. Attempting to install...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if %errorlevel% neq 0 (
        echo Error: Failed to execute uv installation script. Please check your internet connection or PowerShell settings.
        exit /b 1
    )

    rem After installation, update the PATH for the current session.
    echo Installation script finished. Updating PATH for this session...
    set "UV_BIN_PATH=%USERPROFILE%\.local\bin"
    set "PATH=%UV_BIN_PATH%;%PATH%"

    rem Verify again to ensure that both the installation and path settings have taken effect.
    echo Verifying installation...
    uv --version
    if %errorlevel% neq 0 (
        echo Error: uv was installed but is still not working after PATH update.
        echo Please check the installation path is correct: %UV_BIN_PATH%
        exit /b 1
    )

    echo Verification successful!
    exit /b 0

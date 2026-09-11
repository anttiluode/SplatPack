@echo off
setlocal

rem In a repo checkout, splatpack.py can shadow the installed console command
rem on Windows when .PY is in PATHEXT. Prefer the pip-generated .exe, which is
rem bound to the exact Python environment where SplatPack (and NumPy) were installed.
for /f "delims=" %%I in ('where splatpack.exe 2^>nul') do (
  call "%%I" %*
  exit /b
)

rem Fallback for source checkouts without an installed console script. Only use
rem an interpreter that can import NumPy, so we do not accidentally choose a
rem different Python installation.
where python >nul 2>&1 && python -c "import numpy" >nul 2>&1 && goto python
where py >nul 2>&1 && py -3 -c "import numpy" >nul 2>&1 && goto pylauncher
where python3.13 >nul 2>&1 && python3.13 -c "import numpy" >nul 2>&1 && goto py313

echo SplatPack: no usable Python environment found. Run: python -m pip install -e . 1>&2
exit /b 9009

:python
python "%~dp0splatpack.py" %*
exit /b %errorlevel%

:pylauncher
py -3 "%~dp0splatpack.py" %*
exit /b %errorlevel%

:py313
python3.13 "%~dp0splatpack.py" %*
exit /b %errorlevel%

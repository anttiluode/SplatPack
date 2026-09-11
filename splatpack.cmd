@echo off
setlocal

where python3.13 >nul 2>&1 && goto py313
where py >nul 2>&1 && goto pylauncher
where python >nul 2>&1 && goto python

echo SplatPack: no Python interpreter found on PATH. 1>&2
exit /b 9009

:py313
python3.13 "%~dp0splatpack.py" %*
exit /b %errorlevel%

:pylauncher
py -3 "%~dp0splatpack.py" %*
exit /b %errorlevel%

:python
python "%~dp0splatpack.py" %*
exit /b %errorlevel%

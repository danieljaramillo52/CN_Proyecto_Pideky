@echo off
echo Ejecucion proyecto filtros pideky...
echo ===============================

echo Ejecutando la automatizacion...

cd /d "%~dp0"
.\python-3.12.5-emb\python.exe -m streamlit run Scripts\main.py

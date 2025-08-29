@echo off
echo Preparando los insumos de la automatización...
echo ===============================


cd /d "%~dp0"
.\python-3.12.5-emb\python.exe Scripts\transformador_insumos.py

echo Proceso de conversión finalizado cierre la ventana actual ... 
pause 
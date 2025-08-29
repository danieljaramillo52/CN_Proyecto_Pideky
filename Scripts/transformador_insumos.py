from xlsx2csv import Xlsx2csv
from pathlib import Path
from loguru import logger
from os import path, listdir


def export_sheets_with_filename(xlsx_path, outdir):
    """

    Args:
        xlsx_path (str): Ruta relativa completa del archivo .xlsx a transformar
        outdir (str): Directorio de salida para los archivos transformados. 
    """
    # Directorio de salida.
    outdir = Path(outdir)

    # Varificación de existecia sino lo crea.
    outdir.mkdir(exist_ok=True)

    base = Path(xlsx_path).stem  # nombre base del archivo sin extensión

    # Archivo procesado sin cargar en memoria, codificación latin1
    # Codificación para caracteres del español
    x2c = Xlsx2csv(xlsx_path, outputencoding="utf-8")

    # Recorre todas las hojas (extrayendo el nombre de cada una y guardandolo en un artchivo independiente.)
    for sheet_index, sheet_name in enumerate(x2c.workbook.sheets, start=1):
        outname = outdir / f"{base}_{sheet_name['name']}.csv"
        x2c.convert(str(outname), sheetid=sheet_index)
        logger.info(f"✅ Exportado: {outname}")


# Ejecución utilidad conversor. 
if __name__ == "__main__":
    
    DIR_VTAS = "Insumos/maestra_vtas"
    DIR_MA_CLI = "Insumos/maestra_clientes"
    
    rutas_arc_vtas = [path.join(DIR_VTAS, nombre) for nombre in listdir(DIR_VTAS) if ".xlsx" in nombre]
    rutas_maestra_clientes = [path.join(DIR_MA_CLI, nombre) for nombre in listdir(DIR_MA_CLI) if ".xlsx" in nombre]
    
    rutas_completas = rutas_arc_vtas + rutas_maestra_clientes
    
for cada_path in rutas_arc_vtas:
    logger.info(f"Procesando: {cada_path[25:]}")
    xlsx = cada_path
    export_sheets_with_filename(xlsx_path=xlsx, outdir=r"Insumos/transformados")

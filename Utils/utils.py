import pandas as pd
import os
import yaml
import requests
import numpy as np
import streamlit as st
import re
import datetime as dt
import base64
import threading
from datetime import date # advertencia a modulos nativos. Por sobreescribir los del entorno virtual.
from io import BytesIO
from typing import List, Union, Literal ,Sequence, Dict, Any, Sequence, Optional, Tuple
from loguru import logger
from PIL import Image

from Scripts.config_path_routes import ConfigPathRoutes


def procesar_configuracion(nom_archivo_configuracion: str) -> dict:
    """Lee un archivo YAML de configuración para un proyecto.

    Args:
        nom_archivo_configuracion (str): Nombre del archivo YAML que contiene
            la configuración del proyecto.

    Returns:
        dict: Un diccionario con la información de configuración leída del archivo YAML.
    """
    try:
        with open(nom_archivo_configuracion, "r", encoding="utf-8") as archivo:
            configuracion_yaml = yaml.safe_load(archivo)
        logger.success("Proceso de obtención de configuración satisfactorio")
    except Exception as e:
        logger.critical(f"Proceso de lectura de configuración fallido {e}")
        raise e

    return configuracion_yaml

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    """Carga archivos csv desde un path especifico

    Args:
        path (str): Ruta relativa del directorio que queremos procesar. 

    Returns:
        pd.DataFrame: df procesado
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        df = pd.read_csv(f, sep=",", dtype=str)
    return df

@st.cache_data(show_spinner=False)
def lectura_simple_excel(
    dir_insumo: str, nom_insumo: str, nom_hoja: str = None
) -> pd.DataFrame:
    """
    Lee un archivo de Excel y devuelve su contenido en un DataFrame.

    Args:
        dir_insumo (str): Ruta del directorio donde se encuentra el archivo.
        nom_insumo (str): Nombre del archivo de Excel (incluyendo la extensión).
        nom_hoja (str): Nombre de la hoja a leer dentro del archivo de Excel.

    Returns:
        pd.DataFrame: Contenido de la hoja de Excel como un DataFrame.

    Raises:
        Exception: Si ocurre algún error durante la lectura del archivo.
    """

    try:
        logger.info(f"Inicio lectura simple de {nom_insumo}")
        base_leida = pd.read_excel(
            dir_insumo + nom_insumo,
            sheet_name=nom_hoja,
            dtype=str,
        )
        logger.success(f"Lectura simple de {nom_insumo} realizada con éxito")
        return base_leida
    except Exception as e:
        logger.error(f"Proceso de lectura fallido: {e}")
        raise Exception(f"Error al leer el archivo: {e}")

def construir_filtros_desde_df(
    df: pd.DataFrame,
    columnas: Optional[Sequence[str]] = None,
    col_fecha: str = "Fecha",
    rango_fecha: Optional[Tuple[str, str]] = None,
    max_unicos: int = 200,
) -> Dict[str, Any]:
    """
    Genera un dict 'filtros_plantilla' a partir de un DataFrame.

    - Columnas normales  -> lista de valores únicos (excluye nulos y strings vacíos).
    - Columna 'Fecha'    -> tupla (inicio, fin) como 'YYYY-MM-DD'.
      * Si 'rango_fecha' se provee, se usa ese rango.
      * Si no, se toma min/max presentes en el DataFrame.

    Args:
        df: DataFrame base.
        columnas: Subconjunto de columnas a considerar (por defecto, todas).
        col_fecha: Nombre de la columna a tratar como fecha (por defecto "Fecha").
        rango_fecha: Rango explícito (inicio, fin) para la columna de fecha.
        max_unicos: Tope de cardinalidad para no incluir columnas con demasiados únicos.

    Returns:
        Dict[str, Any] con el formato de filtros_plantilla.
    """
    if columnas is None:
        columnas = list(df.columns)

    filtros: Dict[str, Any] = {}

    for col in columnas:
        if col not in df.columns:
            continue

        serie = df[col]

        # Limpia strings vacíos como nulos
        if pd.api.types.is_string_dtype(serie):
            serie = serie.astype("string").str.strip().replace("", pd.NA)

        serie = serie.dropna()
        if serie.empty:
            continue  # omite columnas sin datos

        # Caso especial: columna de fecha
        if col.casefold() == col_fecha.casefold():
            if rango_fecha is not None:
                ini = pd.to_datetime(rango_fecha[0]).date().isoformat()
                fin = pd.to_datetime(rango_fecha[1]).date().isoformat()
            else:
                fechas = pd.to_datetime(serie, errors="coerce").dropna()
                if fechas.empty:
                    continue
                ini = fechas.min().date().isoformat()
                fin = fechas.max().date().isoformat()
            filtros[col] = (ini, fin)
            continue

        # Columnas normales: valores únicos
        unicos = pd.unique(serie)

        # Convierte numpy scalars a Python nativo (serializable)
        valores = []
        for v in unicos:
            if isinstance(v, np.generic):
                v = v.item()
            valores.append(v)

        if len(valores) == 0:
            continue

        if len(valores) > max_unicos:
            # Demasiados únicos -> omitir por defecto (evita plantillas gigantes)
            continue

        # Ordena si es posible
        try:
            valores = sorted(valores)
        except Exception:
            pass

        filtros[col] = list(valores)

    return filtros

def crear_boton_exportar(df, filename="selecciones.xlsx", key=None):
    """
    Crea un botón interactivo en Streamlit para exportar un DataFrame como archivo Excel descargable.

    Esta función genera un botón de descarga que permite al usuario exportar los datos contenidos
    en un DataFrame de pandas directamente desde la interfaz de Streamlit. El archivo se genera
    en memoria sin necesidad de almacenamiento temporal en disco.

    Args:
        df (pd.DataFrame): DataFrame de pandas que contiene los datos a exportar.
        filename (str, opcional): Nombre del archivo a descargar. Debe incluir extensión .xlsx.
                                  Por defecto: "selecciones.xlsx".
        key (str, opcional): Clave única para identificación del elemento en Streamlit. Si no se
                            proporciona, se generará automáticamente basado en el nombre de archivo.
                            Necesario para evitar conflictos cuando existen múltiples botones.

    Returns:
        None: La función no retorna ningún valor, pero renderiza un elemento interactivo en la UI.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'columna': [1, 2, 3]})
        >>> crear_boton_exportar(df, filename="datos.xlsx", key="boton_unic0")

        >>> # Uso con key automático
        >>> crear_boton_exportar(df, filename="reporte_diario.xlsx")

    Note:
        - Para prevenir errores de claves duplicadas en Streamlit, especialmente cuando se usan
        múltiples instancias del botón, es recomendable proveer una clave única mediante el parámetro `key`.
        - El archivo se genera usando openpyxl como motor de Excel, asegurando compatibilidad con
        formatos .xlsx modernos.
        - La función utiliza un buffer en memoria para máxima eficiencia, evitando operaciones de I/O en disco.
    """
    # Crear un buffer en memoria para almacenar el archivo Excel
    output = BytesIO()

    # Guardar el DataFrame en el buffer como un archivo Excel
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Datos")

    # Obtener el contenido del buffer
    excel_data = output.getvalue()

    # Si no se proporciona un key, generar uno basado en el nombre del archivo
    key = key if key is not None else f"download_button_{filename}"

    # Botón de descarga con clave única
    st.download_button(
        label=f"Descargar {filename}",
        data=excel_data,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key,  # Clave única para evitar conflictos
    )


def eliminar_espacios_cols(df, columnas: str | list):
    """
    Elimina todos los espacios en blanco de los valores en las columnas
    especificadas de un DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame que contiene las columnas a limpiar.
        columnas (str o list): Nombre de la columna o lista de nombres de
                            columnas en las que se eliminarán los espacios
                            en blanco.

    Returns:
        pd.DataFrame: El DataFrame con los espacios en blanco eliminados de las columnas
                    especificadas.
    """
    # Si columnas es un solo nombre, convertirlo a lista
    if isinstance(columnas, str):
        columnas = [columnas]
    # Aplicar str.strip() solo a las columnas especificadas
    for columna in columnas:
        if columna in df.columns and df[columna].dtype == "object":
            df.loc[:, columna] = df[columna].str.replace(" ", "")
        else:
            print(f"La columna '{columna}' no existe o no es de tipo string.")
    return df


# Cargar el archivo CSS
def load_css(file_name):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Archivo CSS no encontrado: {file_name}")
    except Exception as e:
        st.error(f"Error al cargar el CSS: {e}")


def image_to_base64(image):
    """Convierte una imagen PIL a base64 para HTML inline"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def setup_ui():
    # Si tienes estilos base

    rutas = ConfigPathRoutes()
    ruta_logo = rutas.resolver_ruta("Img", "logo_pideky.jpg")

    load_css("static/styles.css")
    # Cargar imagen como base64 para incrustarla en HTML
    image = Image.open(ruta_logo).resize((300, 250))
    img_base64 = image_to_base64(image)

    # Usar una tabla HTML para alinear horizontalmente texto e imagen
    st.markdown(
        f"""
        <div style="background-color: #e8f8f5; padding: 10px 20px; border-radius: 10px; autocomplete="off;">
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="text-align: left;">
                        <h1 style="margin: 0; font-size: 2.5em; color: ##9bb80b;">
                            Iniciativas Pideky
                        </h1>
                    </td>
                    <td style="text-align: right; vertical-align: middle;">
                        <img src="data:image/png;base64,{img_base64}" alt="Logo Pideky" width="100">
                    </td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


# wrapper: Decorador (st.cache data.)
# Permite almacenar el resultado del método en cache.
# Así evitamos multiples llamadas a la API y consultas a los archivos en google shets.
@st.cache_data
def fetch_data_from_url(url: str | None) -> pd.DataFrame:
    """
    Obtiene datos JSON desde una URL y los convierte en un DataFrame de pandas.

    Args:
        url (str): La URL desde donde se obtendrán los datos JSON.

    Returns:
        JSON: Un elemento JSON con los datos obtenidos

    ¿Qué hace requests.get(url)?
        Realiza una solicitud HTTP GET:
            - Una solicitud GET es un tipo de solicitud HTTP que se utiliza para obtener datos de un servidor.

            - Cuando llamas a requests.get(url), se envía una solicitud GET a la URL proporcionada.

        Recibe una respuesta:
         El servidor procesa la solicitud y devuelve una respuesta. Esta respuesta incluye:
            - Un código de estado HTTP (por ejemplo, 200 para éxito, 404 para no encontrado, 500 para error del servidor, etc.).

            - Encabezados (metadata sobre la respuesta).

            - Cuerpo (los datos devueltos, como HTML, JSON, XML, etc.).

        Devuelve un objeto Response:
            - requests.get(url) devuelve un objeto de tipo Response, que contiene toda la información de la respuesta.

            - Puedes acceder a los datos de la respuesta usando los atributos y métodos de este objeto.
    """

    with requests.get(url) as response:
        if response.status_code == 200:
            data = response.json()
            return data  # Convertir JSON a DataFrame
        else:
            raise Exception(f"Error al obtener datos: {response.status_code}")


def lectura_auxiliares_css_js(nom_modulo: str, encoding: str = "utf-8"):
    """Procesa los archivos auxiliares tipo .css y .js para la modificación de estilos de la interfaz.

    Args:
        nom_modulo (str): Nombre del módulo dentro de la carpeta "static".
        encoding (str, optional): Codificación al abrir el archivo correspondiente. Defaults to "utf-8".

    Exceptions:
        FileNotFoundError: Si el archivo no existe en la ruta especificada.
        UnicodeDecodeError: Si hay un problema con la codificación del archivo.
        Exception: Captura cualquier otro error inesperado.
    """
    try:
        script_path = os.path.join("static", nom_modulo)

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"El archivo '{script_path}' no fue encontrado.")

        with open(script_path, "r", encoding=encoding) as f:
            script_content = f.read()

        st.markdown(f"<script>{script_content}</script>", unsafe_allow_html=True)

    except FileNotFoundError as e:
        st.error(f"Error: {e}")
    except UnicodeDecodeError:
        st.error(
            f"Error: No se pudo leer el archivo '{script_path}' debido a un problema de codificación."
        )
    except Exception as e:
        st.error(f"Error inesperado: {e}")


def json_a_dataframe(data):
    """
    Convierte una estructura de datos JSON en un DataFrame de pandas,
    usando la primera fila como los encabezados y estableciendo el tipo de datos a cadenas de texto (str).

    Parameters:
    data (list): Lista de listas donde la primera lista contiene los encabezados y el resto contiene los datos.

    Returns:
    DataFrame: Un DataFrame de pandas con los datos proporcionados, con las columnas establecidas
    según la primera fila y el tipo de datos de todas las columnas como cadenas de texto (str).

    Raises:
    ValueError: Si el JSON no tiene al menos dos filas (una para los encabezados y una para los datos).
    """
    try:
        if len(data) < 2:
            raise ValueError(
                "El JSON debe contener al menos dos filas: una para los encabezados y una para los datos."
            )

        df = pd.DataFrame(data[1:], columns=data[0], dtype=str)
        return df

    except ValueError as ve:
        logger.error(f"Error: {ve}")
    except Exception as e:
        logger.error(f"Ocurrió un error inesperado: {e}")




def left_merge_on_columns(
    df1: pd.DataFrame, df2: pd.DataFrame, key_columns: list
) -> pd.DataFrame:
    """
    Realiza un left merge entre dos DataFrames usando una lista de columnas comunes como llave.

    Parámetros:
    -----------
    df1 : pd.DataFrame
        DataFrame base sobre el cual se hará el merge.
    df2 : pd.DataFrame
        DataFrame que se unirá a df1 basado en las columnas especificadas.
    key_columns : list
        Lista de nombres de columnas en las que se basará la fusión.

    Retorna:
    --------
    pd.DataFrame
        Un nuevo DataFrame con la combinación de df1 y df2, manteniendo todas las filas de df1.
    """
    # Validar que todas las columnas clave existen en ambos DataFrames
    for col in key_columns:
        if col not in df1.columns:
            raise KeyError(f"La columna '{col}' no está en df1")
        if col not in df2.columns:
            raise KeyError(f"La columna '{col}' no está en df2")

    # Realizar el left merge
    merged_df = df1.merge(df2, on=key_columns, how="left")

    return merged_df





def calcular_totales(df, porcentaje_crecimiento):
    """
    Calcula el crecimiento y total de unidades.

    Args:
        df (pd.DataFrame): DataFrame con columna 'Unidades'.
        porcentaje_crecimiento (float): Porcentaje de crecimiento a aplicar.

    Returns:
        pd.DataFrame: DataFrame con columnas 'Crec actividad' y 'unidades_totales'.
    """
    df["Crec actividad"] = df["Unidades"] * porcentaje_crecimiento / 100
    df["unidades_totales"] = np.ceil(df["Unidades"] + df["Crec actividad"])

    df["Crec actividad"] = df["Crec actividad"].astype(int)
    return df

def concatenar_dataframes(df_list: list[pd.DataFrame]):
        """
        Concatena una lista de DataFrames.

        Args:
            df_list: Lista de DataFrames a concatenar.

        Returns:
            Un DataFrame concatenado.
        """
        try:
            if len(df_list) != 1:
                #logger.info("Inicio concatenacion de dataframes")
                concatenados = pd.concat(df_list, ignore_index=True, join="inner")
                #logger.success("se concatenaron los dataframes correctamente")
                return concatenados
            else:
                return df_list[0]
        except Exception as e:
            logger.critical(e)
            raise e
        


def obtener_rango_valido_desde_texto(
    texto: str, por_defecto: tuple[int, int] = (5, 10)
) -> tuple[int, int]:
    """
    Extrae un rango de dos números enteros desde un texto. Si no se encuentran dos números, retorna un rango por defecto.

    Args:
        texto (str): Texto que contiene números, por ejemplo "5% - 10%".
        por_defecto (tuple[int, int]): Rango de retorno si falla la extracción.

    Returns:
        tuple[int, int]: Rango de dos valores enteros.
    """
    try:
        numeros = tuple(map(int, re.findall(r"\d+", texto)))
        if len(numeros) < 2:
            return por_defecto
        return numeros
    except Exception:
        return por_defecto



def aplanar_diccionario(diccionario: dict, clave_aplanar: str = "Fecha") -> dict:
    """
    Aplana un diccionario anidado moviendo las claves de un subdiccionario especificado al nivel superior.

    Parámetros:
        diccionario (dict): Diccionario original con posible anidación
        clave_aplanar (str): Clave del subdiccionario a aplanar (default: "Fecha")

    Retorno:
        dict: Diccionario aplanado con todas las claves al nivel superior

    Ejemplo:
        >>> dict_original = {"a": 1, "Fecha": {"mes": "Enero", "año": 2023}}
        >>> aplanar_diccionario(dict_original)
        {"a": 1, "mes": "Enero", "año": 2023}
    """
    return {
        **{k: v for k, v in diccionario.items() if k != clave_aplanar},
        **diccionario.get(clave_aplanar, {}),
    }


def renombrar_columnas_con_diccionario(
    df: pd.DataFrame, cols_to_rename: dict
) -> pd.DataFrame:
    """Funcion que toma un diccionario con keys ( nombres actuales ) y values (nuevos nombres) para remplazar nombres de columnas en un dataframe.
    Args:
        base: dataframe al cual se le harán los remplazos
        cols_to_rename: diccionario con nombres antiguos y nuevos
    Result:
        base_renombrada: Base con las columnas renombradas.
    """
    base_renombrada = None

    try:
        base_renombrada = df.rename(columns=cols_to_rename, inplace=False)
        # logger.success("Proceso de renombrar columnas satisfactorio: ")
    except Exception:
        logger.critical("Proceso de renombrar columnas fallido.")
        raise Exception

    return base_renombrada


def reemplazar_columna_en_funcion_de_otra(
    df: pd.DataFrame,
    nom_columna_a_reemplazar: str,
    nom_columna_de_referencia: str,
    mapeo: dict,
) -> pd.DataFrame:
    """
    Reemplaza los valores en una columna en función de los valores en otra columna en un DataFrame.

    Args:
        df (pandas.DataFrame): El DataFrame en el que se realizarán los reemplazos.
        columna_a_reemplazar (str): El nombre de la columna que se reemplazará.
        columna_de_referencia (str): El nombre de la columna que se utilizará como referencia para el reemplazo.
        mapeo (dict): Un diccionario que mapea los valores de la columna de referencia a los nuevos valores.

    Returns:
        pandas.DataFrame: El DataFrame actualizado con los valores reemplazados en la columna indicada.
    """
    try:
        # logger.info(
        #    f"Inicio de remplazamiento de datos en {nom_columna_a_reemplazar}"
        # )
        df.loc[:, nom_columna_a_reemplazar] = np.where(
            df[nom_columna_de_referencia].isin(mapeo.keys()),
            df[nom_columna_de_referencia].map(mapeo),
            df[nom_columna_a_reemplazar],
        )
        logger.success(
            f"Proceso de remplazamiento en {nom_columna_a_reemplazar} exitoso"
        )
    except Exception as e:
        logger.critical(
            f"Proceso de remplazamiento de datos en {nom_columna_a_reemplazar} fallido."
        )
        raise e

    return df


@staticmethod
def Seleccionar_columnas_pd(
    df: pd.DataFrame, cols_elegidas: List[str]
) -> pd.DataFrame | None:
    """
    Filtra y retorna las columnas especificadas del DataFrame.

    Parámetros:
    dataframe (pd.DataFrame): DataFrame del cual se filtrarán las columnas.
    cols_elegidas (list): Lista de nombres de las columnas a incluir en el DataFrame filtrado.

    Retorna:
    pd.DataFrame: DataFrame con las columnas filtradas.
    """
    try:
        # Verificar si dataframe es un DataFrame de pandas
        if not isinstance(df, pd.DataFrame):
            raise TypeError("El argumento 'dataframe' debe ser un DataFrame de pandas.")

        # Filtrar las columnas especificadas
        df_filtrado = df[cols_elegidas]

        # Registrar el proceso
        logger.info(f"Columnas filtradas: {', '.join(cols_elegidas)}")

        return df_filtrado

    except KeyError as ke:
        error_message = (
            f"Error: Columnas especificadas no encontradas en el DataFrame: {str(ke)}"
        )
        return df
    except Exception as e:
        logger.critical(f"Error inesperado al filtrar columnas: {str(e)}")


def group_by_and_operate(
    df: pd.DataFrame,
    group_col: Union[str, List[str]],
    operation_cols: Union[str, List[str]],
    operation: Literal["sum", "mean", "count"] = "sum",
) -> pd.DataFrame:
    """
    Agrupa un DataFrame por una o varias columnas y aplica una operación sobre otras columnas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada que contiene los datos a procesar.
    group_col : str or list of str
        Columna o lista de columnas por las que se agrupará el DataFrame.
    operation_cols : str or list of str
        Columna o columnas sobre las que se aplicará la operación (sum, mean, count).
    operation : {'sum', 'mean', 'count'}, default='sum'
        Operación a realizar sobre las columnas indicadas.

    Returns
    -------
    pd.DataFrame or None
        DataFrame resultante con los valores agrupados y operados, o `None` si ocurre un error.
    """
    try:
        group_keys = [group_col] if isinstance(group_col, str) else group_col
        target_cols = (
            [operation_cols] if isinstance(operation_cols, str) else operation_cols
        )

        if operation == "sum":
            result_df = df.groupby(group_keys)[target_cols].sum().reset_index()
        elif operation == "mean":
            result_df = df.groupby(group_keys)[target_cols].mean().reset_index()
        elif operation == "count":
            result_df = df.groupby(group_keys)[target_cols].count().reset_index()
        else:
            raise ValueError(f"Operación no soportada: '{operation}'")

        logger.info(f"Agrupación y operación '{operation}' realizadas con éxito.")
        return result_df

    except Exception as e:
        logger.critical(f"Error al realizar la operación '{operation}': {e}")
        return df

def filtrar_por_valores(
    df: pd.DataFrame, columna: str, valores: list[str | int], incluir: bool = True
) -> pd.DataFrame | None:
    """
    Filtra un DataFrame incluyendo o excluyendo filas según los valores en una columna.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a filtrar.
    columna : str
        Nombre de la columna sobre la cual aplicar el filtro.
    valores : list of str or int
        Lista de valores a incluir o excluir.
    incluir : bool, default=True
        Si True, incluye las filas con los valores indicados. Si False, las excluye.

    Returns
    -------
    pd.DataFrame or None
        DataFrame filtrado o None si ocurre un error.
    """
    try:
        if isinstance(valores, (str, int)):
            valores = [valores]

        if incluir:
            df_filtrado = df[df[columna].isin(valores)]
        else:
            df_filtrado = df[~df[columna].isin(valores)]

        return df_filtrado

    except Exception as e:
        logger.critical(f"Error al filtrar por valores en la columna '{columna}': {e}")
        return None

def Filtrar_df_dict_clave_valor(df, filtros):
        """
        Filtra el DataFrame basado en un diccionario de condiciones.
        Cada condición puede incluir múltiples valores posibles para cada columna.

        Args:
        df (pd.DataFrame): DataFrame a filtrar.
        filtros (dict): Diccionario con las columnas y los valores (lista) a filtrar.

        Returns:
        pd.DataFrame: DataFrame filtrado.
        """
        mask = pd.Series([True] * len(df))
        for columna, valores in filtros.items():
            if isinstance(valores, list):
                mask &= df[columna].isin(valores)
            else:
                mask &= df[columna] == valores
        return df[mask]

def concatenar_columnas_pd(
    dataframe: pd.DataFrame,
    cols_elegidas: List[str],
    nueva_columna: str,
    sep: str = "",
    omitir_vacios: bool = False,
) -> pd.DataFrame:
    """
    Concatena las columnas especificadas y agrega el resultado como una nueva columna.
    
    Parámetros:
        dataframe: DataFrame origen.
        cols_elegidas: Columnas a concatenar (en orden).
        nueva_columna: Nombre de la columna de salida.
        sep: Separador entre valores concatenados (por defecto: "").
        omitir_vacios: Si True, no inserta separador para valores vacíos/NaN.
    """
    try:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("El argumento 'dataframe' debe ser un DataFrame de pandas.")

        faltantes = [c for c in cols_elegidas if c not in dataframe.columns]
        if faltantes:
            raise KeyError(f"Columnas inexistentes: {', '.join(faltantes)}")

        df = dataframe.copy()

        if omitir_vacios:
            # Convierte a str, quita NaN -> "", y une solo los que no están vacíos
            df[nueva_columna] = (
                df[cols_elegidas]
                .astype(str)
                .fillna("")
                .agg(lambda row: sep.join([v for v in row if v != ""]), axis=1)
            )
        else:
            # Junta tal cual, respetando posiciones (vacíos generan separadores consecutivos)
            df[nueva_columna] = (
                df[cols_elegidas].astype(str).fillna("").agg(sep.join, axis=1)
            )

        logger.info(
            f"Columnas '{', '.join(cols_elegidas)}' concatenadas en '{nueva_columna}' con sep='{sep}' (omitir_vacios={omitir_vacios})."
        )
        return df

    except Exception as e:
        logger.critical(f"Error inesperado al concatenar columnas: {e}")
        return None
    

KeyArg = Union[str, Sequence[str]]

def pd_left_merge_two_keys(
    base_left: pd.DataFrame,
    base_right: pd.DataFrame,
    left_key: KeyArg,
    right_key: KeyArg,
    conservar_llave_derecha: bool = False,
    suffixes: tuple[str, str] = ("", "_r"),
) -> pd.DataFrame:
    """
    Realiza un left join entre dos DataFrames permitiendo llaves simples o compuestas.

    - Acepta `left_key` y `right_key` como `str` o `list[str]`.
    - Si las llaves izquierda y derecha son idénticas en nombre y orden, usa `on=` para
      evitar duplicados de columnas clave; en caso contrario usa `left_on`/`right_on`.
    - Si `conservar_llave_derecha` es False, elimina las columnas de la(s) llave(s) derecha(s)
      resultantes del merge (cuando correspondan).

    Args:
        base_left: DataFrame base (lado izquierdo).
        base_right: DataFrame a unir (lado derecho).
        left_key: Columna o lista de columnas del izquierdo.
        right_key: Columna o lista de columnas del derecho.
        conservar_llave_derecha: Si False, elimina las columnas de `right_key` tras el merge.
        suffixes: Sufijos para columnas solapadas (no claves).

    Returns:
        DataFrame resultante del left join.
    """
    # Validaciones básicas
    if not isinstance(base_left, pd.DataFrame):
        raise ValueError("base_left debe ser un DataFrame")
    if not isinstance(base_right, pd.DataFrame):
        raise ValueError("base_right debe ser un DataFrame")

    # Normalizar a listas
    left_keys = [left_key] if isinstance(left_key, str) else list(left_key)
    right_keys = [right_key] if isinstance(right_key, str) else list(right_key)

    if len(left_keys) != len(right_keys):
        raise ValueError(
            f"Las cantidades de llaves no coinciden: left={len(left_keys)} vs right={len(right_keys)}"
        )

    # Comprobar existencia de columnas
    faltantes_left = [c for c in left_keys if c not in base_left.columns]
    faltantes_right = [c for c in right_keys if c not in base_right.columns]
    if faltantes_left:
        raise KeyError(f"Columnas faltantes en izquierdo: {faltantes_left}")
    if faltantes_right:
        raise KeyError(f"Columnas faltantes en derecho: {faltantes_right}")

    try:
        # Si las llaves son idénticas (mismos nombres y orden), usar `on=`
        if left_keys == right_keys:
            base = pd.merge(
                base_left,
                base_right,
                how="left",
                on=left_keys if len(left_keys) > 1 else left_keys[0],
                suffixes=suffixes,
            )
        else:
            base = pd.merge(
                base_left,
                base_right,
                how="left",
                left_on=left_keys if len(left_keys) > 1 else left_keys[0],
                right_on=right_keys if len(right_keys) > 1 else right_keys[0],
                suffixes=suffixes,
            )

        logger.success("Proceso de merge satisfactorio")

        # Eliminar llaves derechas si se usó left_on/right_on y así se solicita
        if not conservar_llave_derecha and left_keys != right_keys:
            cols_a_borrar = [c for c in right_keys if c in base.columns]
            if cols_a_borrar:
                base = base.drop(columns=cols_a_borrar)

    except pd.errors.MergeError as e:
        logger.critical(f"Proceso de merge fallido: {e}")
        raise

    return base


from datetime import datetime
from io import BytesIO
import streamlit as st
import st_file_uploader as stf
import pandas as pd
from datetime import datetime
from typing import Any, Optional, Tuple, List, Union


def add_key_ss_st(clave: str, valor_inicial: Any) -> None:
    """
    Inicializa una clave en st.session_state con un valor por defecto si no existe.

    Args:
        clave (str): Nombre de la variable a inicializar en session_state.
        valor_inicial (Any): Valor por defecto que se asignará si la clave no existe.

    Returns:
        None

    Raises:
        TypeError: Si 'clave' no es una cadena de texto.
        RuntimeError: Si st.session_state no está disponible en el entorno actual.
    """
    try:
        if not isinstance(clave, str):
            raise TypeError("La clave debe ser de tipo str.")

        if not hasattr(st, "session_state"):
            raise RuntimeError("st.session_state no está disponible.")

        if clave not in st.session_state:
            st.session_state[clave] = valor_inicial

    except Exception as e:
        st.error(f"Error al inicializar '{clave}': {e}")


def set_key_ss_st(clave: str, valor: Any) -> None:
    """
    Asigna o actualiza un valor en st.session_state.

    Args:
        clave (str): Nombre de la variable en session_state.
        valor (Any): Valor que se asignará.
    """
    if not isinstance(clave, str):
        raise TypeError("La clave debe ser de tipo str.")
    st.session_state[clave] = valor


def set_multiple_keys(valores: dict):
    """
    Asigna múltiples valores al session_state usando set_key_ss_st internamente.

    Args:
        valores (dict): Diccionario de clave: valor a asignar.
    """
    for clave, valor in valores.items():
        set_key_ss_st(clave, valor)


def clean_key_ss_st(keys: tuple | list) -> None:
    """
    Elimina de forma segura múltiples claves del estado de sesión de Streamlit.

    Parámetros:
        keys (tuple|list): Iterable con los nombres de las claves a eliminar.

    Ejemplo:
        >>> clean_key_ss_st(("df_procesado", "promedios"))
        >>> clean_key_ss_st(["datos_temp", "filtros"])

    Notas:
        - Elimina claves existentes sin lanzar errores si no existen
        - Registra errores inesperados sin interrumpir el flujo
    """
    try:
        for key in keys:
            st.session_state.pop(key, None)
    except TypeError as e:
        st.error(f"Error en tipo de dato: {str(e)} - Las claves deben ser iterables")
    except Exception as e:
        st.error(f"Error inesperado al limpiar estado: {str(e)}")
        raise


class TextInputManager:
    """
    Maneja una caja de texto en Streamlit con validación numérica,
    advertencias y ubicación configurable.
    """

    def __init__(
        self,
        clave: str,
        etiqueta: str,
        valor_por_defecto: str = "",
        usar_sidebar: bool = True,
        tipo: int | float = float,
        minimo: Optional[float] = None,
        maximo: Optional[float] = None,
        mensaje_error: Optional[str] = None,
    ):
        """
        Inicializa el gestor del campo de texto.

        Args:
            clave (str): Clave única para session_state y key del widget.
            etiqueta (str): Etiqueta visible junto a la caja de texto.
            valor_por_defecto (str): Texto inicial en la caja.
            usar_sidebar (bool): Si True, se muestra en el sidebar; de lo contrario, en el cuerpo principal.
            tipo (type): Tipo numérico esperado (int o float).
            minimo (float): Valor mínimo permitido.
            maximo (float): Valor máximo permitido.
            mensaje_error (str): Mensaje personalizado para mostrar si hay error.
        """
        self.clave = clave
        self.etiqueta = etiqueta
        self.usar_sidebar = usar_sidebar
        self.valor_por_defecto = valor_por_defecto
        self.tipo = tipo
        self.minimo = minimo
        self.maximo = maximo
        self.mensaje_error = (
            mensaje_error or f"Ingrese un número válido entre {minimo} y {maximo}."
        )

        self._show_input()

    def _show_input(self):
        # Selecciona el contenedor (sidebar o cuerpo principal)
        contenedor = st.sidebar if self.usar_sidebar else st

        input_func = contenedor.text_input

        valor_ingresado = input_func(
            label=self.etiqueta, value=self.valor_por_defecto, key=self.clave
        )

        st.session_state[f"{self.clave}_valido"] = False

        if self.tipo == str:
            if valor_ingresado.strip():
                st.session_state[f"{self.clave}_valido"] = True
            else:
                contenedor.warning("Este campo no puede estar vacío.")
        else:
            try:
                valor_numerico = self.tipo(valor_ingresado)

                if (self.minimo is not None and valor_numerico < self.minimo) or (
                    self.maximo is not None and valor_numerico > self.maximo
                ):
                    contenedor.warning(self.mensaje_error)
                else:
                    st.session_state[f"{self.clave}_valido"] = True
            except ValueError:
                if valor_ingresado.strip():
                    contenedor.warning("Debe ingresar un número válido.")

    def get_value(self) -> Optional[Union[int, float, str]]:
        """
        Retorna el valor ingresado si es válido, convertido según el tipo especificado.

        Returns:
            int, float, str o None
        """
        if not st.session_state.get(f"{self.clave}_valido", False):
            return None

        try:
            return (
                self.tipo(st.session_state[self.clave])
                if self.tipo != str
                else st.session_state[self.clave]
            )
        except:
            return None

    def is_valid(self) -> bool:
        """
        Indica si el valor ingresado es numérico y está dentro del rango.

        Returns:
            bool
        """
        return st.session_state.get(f"{self.clave}_valido", False)

    def reset(self) -> None:
        """
        Marca el text_input para reiniciarse en la próxima ejecución.

        Nota: No modifica session_state directamente si el widget ya fue instanciado.
        """
        st.session_state[f"{self.clave}_reset"] = True


class MultiVisibilityController:
    """
    Controlador de visibilidad para múltiples componentes en Streamlit.

    Esta clase permite gestionar de forma centralizada la visibilidad de varios elementos
    de la interfaz (como inputs, selectboxes, sliders, etc.) usando claves en `st.session_state`.

    Attributes:
        claves (list[str]): Lista de claves de visibilidad asociadas a cada componente.

    Methods:
        mostrar(): Establece todas las claves como visibles (`True`).
        ocultar(): Establece todas las claves como no visibles (`False`).
        esta_visible() -> bool: Retorna `True` si todos los elementos están visibles.
        mostrar_clave(clave: str): Establece como visible un componente específico.
        ocultar_clave(clave: str): Establece como no visible un componente específico.
        esta_visible_clave(clave: str) -> bool: Verifica si un componente está visible.
    """

    def __init__(self, claves: list[str], visibles_por_defecto: bool = True):
        """
        Inicializa el controlador, estableciendo el estado inicial de visibilidad
        para cada clave proporcionada.

        Args:
            claves (list[str]): Lista de claves para controlar la visibilidad.
            visibles_por_defecto (bool): Estado inicial de visibilidad (True = mostrar).
        """
        self.claves = claves
        self.claves = [f"{clave}_visible" for clave in claves]
        for clave in self.claves:
            if clave not in st.session_state:
                st.session_state[clave] = visibles_por_defecto

    def mostrar(self):
        """Muestra todos los componentes controlados (establece todas las claves como True)."""
        for clave in self.claves:
            st.session_state[clave] = True

    def ocultar(self):
        """Oculta todos los componentes controlados (establece todas las claves como False)."""
        for clave in self.claves:
            st.session_state[clave] = False

    def esta_visible(self) -> bool:
        """
        Verifica si todos los elementos controlados están actualmente visibles.

        Returns:
            bool: True si todas las claves están en estado `True`, False en caso contrario.
        """
        return all(st.session_state[clave] for clave in self.claves)

    def mostrar_clave(self, clave: str):
        """
        Muestra un componente específico controlado por su clave.

        Args:
            clave (str): Clave del componente a mostrar.
        """
        vis_key = f"{clave}_visible"
        if vis_key in self.claves:
            st.session_state[vis_key] = True

    def ocultar_clave(self, clave: str):
        """
        Oculta un componente específico controlado por su clave.

        Args:
            clave (str): Clave del componente a ocultar.
        """
        vis_key = f"{clave}_visible"
        if vis_key in self.claves:
            st.session_state[vis_key] = False

    def esta_visible_clave(self, clave: str) -> bool:
        """
        Verifica si un componente específico está visible.

        Args:
            clave (str): Clave del componente a verificar.

        Returns:
            bool: True si la clave está en `True`, False en caso contrario.
        """
        vis_key = f"{clave}_visible"
        return st.session_state.get(vis_key, False)
    

@st.cache_data
def leer_archivo_cacheado(
    bytes_archivo: bytes, tipo: str, usecols=None
) -> pd.DataFrame:
    """
    Carga un archivo en memoria desde su contenido en bytes y lo transforma en un objeto de pandas.

    Esta función se encarga exclusivamente de interpretar archivos subidos como objetos binarios, devolviendo
    su contenido estructurado como DataFrame (para .csv y .xlsx). El resultado
    se cachea para evitar reprocesamiento redundante.

    Args:
        bytes_archivo (bytes): Contenido binario del archivo.
        tipo (str): Tipo de archivo. Acepta: 'csv', 'xlsx'
        usecols (list or None, optional): Columnas a usar en archivos Excel.

    Returns:
        Union[pd.DataFrame, Document, pd.DataFrame]: Objeto correspondiente según tipo de archivo.
            - pd.DataFrame para archivos .csv o .xlsx.
            - DataFrame vacío si el tipo no es reconocido.

    Notes:
        Utiliza BytesIO para convertir los bytes en un buffer de archivo en memoria, compatible con pandas.
    """
    buffer = BytesIO(
        bytes_archivo
    )  # Convierte los bytes en un archivo virtual en memoria (tipo archivo)

    if tipo == "csv":
        return pd.read_csv(buffer)
    elif tipo == "xlsx":
        return pd.read_excel(buffer, dtype=str, engine="openpyxl", usecols=usecols)
    else:
        return pd.DataFrame()


class SelectBoxManager:
    """
    Maneja un selectbox en Streamlit con soporte para placeholder,
    persistencia en session_state, validación, y reinicio controlado.
    """

    SELECCION_INVALIDA = "Seleccione una opción..."

    def __init__(
        self,
        clave: str,
        etiqueta: str,
        opciones: List[str],
        placeholder: str = SELECCION_INVALIDA,
        usar_sidebar: bool = True,
    ):
        """
        Inicializa el gestor del selectbox.

        Args:
            clave (str): Clave única para session_state y key del widget.
            etiqueta (str): Texto que se muestra sobre el selectbox.
            opciones (List[str]): Lista de opciones válidas.
            placeholder (str): Valor por defecto no seleccionable.
            usar_sidebar (bool): Si True, se muestra en la barra lateral; de lo contrario, en el cuerpo principal.
        """
        if not opciones or not all(isinstance(op, str) for op in opciones):
            raise ValueError("Las opciones deben ser una lista de strings no vacía.")

        self.clave = clave
        self.etiqueta = etiqueta
        self.opciones = opciones
        self.placeholder = placeholder
        self.usar_sidebar = usar_sidebar

        self._show_selectbox()

    def _show_selectbox(self):
        opciones_mostradas = [self.placeholder] + self.opciones

        # Si se solicitó un reset previamente, aplicar ahora
        if st.session_state.get(f"{self.clave}_reset", False):
            st.session_state[self.clave] = self.placeholder
            st.session_state[f"{self.clave}_reset"] = False

        # Inicializa si aún no existe
        if self.clave not in st.session_state:
            st.session_state[self.clave] = self.placeholder

        # Mostrar el selectbox con una clave explícita
        if self.usar_sidebar:
            st.sidebar.selectbox(
                label=self.etiqueta,
                options=opciones_mostradas,
                index=opciones_mostradas.index(st.session_state[self.clave]),
                key=self.clave,
            )
        else:
            st.selectbox(
                label=self.etiqueta,
                options=opciones_mostradas,
                index=opciones_mostradas.index(st.session_state[self.clave]),
                key=self.clave,
            )

        # Guarda internamente el valor actual
        self.clave_actual_menu = st.session_state[self.clave]

    def get_value(self) -> str:
        """Devuelve la opción actualmente seleccionada."""
        return st.session_state.get(self.clave, self.clave_actual_menu)

    def is_valid(self) -> bool:
        """Indica si la selección es válida (distinta del placeholder)."""
        return st.session_state.get(self.clave) in self.opciones

    def reset(self) -> None:
        """
        Marca el selectbox para reiniciarse al placeholder en la próxima ejecución.

        Nota: No modifica session_state directamente si el widget ya fue instanciado.
        """
        st.session_state[f"{self.clave}_reset"] = True


class SliderManager:
    """
    Maneja un slider en Streamlit con soporte para persistencia en session_state,
    ubicación en el sidebar o en el cuerpo principal, y validación de rango.
    """

    def __init__(
        self,
        clave: str,
        etiqueta: str,
        rango: Tuple[int, int],
        valor_inicial: Optional[int] = None,
        usar_sidebar: bool = True,
        paso: int = 1,
    ):
        """
        Inicializa el gestor del slider.

        Args:
            clave (str): Clave para session_state.
            etiqueta (str): Texto que se muestra sobre el slider.
            rango (Tuple[int, int]): Rango mínimo y máximo permitido.
            valor_inicial (int, opcional): Valor inicial del slider. Si no se da, se toma el mínimo.
            usar_sidebar (bool): Si True, se muestra en la barra lateral.
            paso (int): Incremento entre valores del slider.
        """
        if not isinstance(rango, tuple) or len(rango) != 2:
            raise ValueError("El rango debe ser una tupla de dos enteros (min, max).")
        if rango[0] >= rango[1]:
            raise ValueError("El valor mínimo del rango debe ser menor que el máximo.")

        self.clave = clave
        self.etiqueta = etiqueta
        self.rango = rango
        self.valor_inicial = valor_inicial if valor_inicial is not None else rango[0]
        self.usar_sidebar = usar_sidebar
        self.paso = paso

        self._mostrar_slider()

    def _mostrar_slider(self):
        if self.clave not in st.session_state:
            st.session_state[self.clave] = self.valor_inicial

        if self.usar_sidebar:
            valor = st.sidebar.slider(
                label=self.etiqueta,
                min_value=self.rango[0],
                max_value=self.rango[1],
                value=st.session_state[self.clave],
                step=self.paso,
            )
        else:
            valor = st.slider(
                self.etiqueta,
                min_value=self.rango[0],
                max_value=self.rango[1],
                value=st.session_state[self.clave],
                step=self.paso,
            )

        st.session_state[self.clave] = valor

    def get(self) -> int:
        """Devuelve el valor actual del slider."""
        return st.session_state[self.clave]

    def is_valid(self) -> bool:
        """Valida si el valor actual está dentro del rango definido."""
        valor = st.session_state[self.clave]
        return self.rango[0] <= valor <= self.rango[1]

    def reset(self) -> None:
        """Reinicia el slider a su valor inicial."""
        st.session_state[self.clave] = self.valor_inicial


class ButtonTracker:
    """
    Gestiona el estado persistente de un botón en Streamlit usando session_state.

    Esta clase permite manejar si un botón ha sido presionado durante la sesión,
    sin entrar en conflicto con las restricciones de Streamlit respecto al uso de claves.
    """

    def __init__(
        self,
        clave: str,
        etiqueta: str = "Enviar",
        usar_sidebar: bool = False,
        auto_render: bool = True,
    ):
        """
        Inicializa el botón y su estado asociado en session_state.

        Args:
            clave (str): Identificador lógico único del botón para el seguimiento de su estado.
            etiqueta (str): Texto visible que se muestra en el botón.
            usar_sidebar (bool): Si es True, el botón se renderiza en la barra lateral;
                                 en caso contrario, en el cuerpo principal.
        """
        self._ph = st.sidebar.empty() if usar_sidebar else st.empty() 
        self.clave_estado = (
            f"{clave}_estado"  # Clave para el estado del botón (presionado o no)
        )
        self.clave_widget = f"{clave}_widget"  # Clave para el widget visual del botón
        self.etiqueta = etiqueta
        self.usar_sidebar = usar_sidebar
        self.auto_render = auto_render
        self._inicializar_estado()

        if self.auto_render:
            self._mostrar_boton()

    def _inicializar_estado(self):
        """
        Inicializa la clave de estado en session_state si aún no existe.

        Este valor es usado para determinar si el botón fue presionado.
        """
        if self.clave_estado not in st.session_state:
            st.session_state[self.clave_estado] = False

    def _mostrar_boton(self):
        """
        Renderiza el botón en la ubicación especificada.

        Si el botón es presionado por el usuario, actualiza su estado lógico
        en session_state a True para permitir su consulta posterior.
        """
        presionado = (
            st.sidebar.button(label=self.etiqueta, key=self.clave_widget)
            if self.usar_sidebar
            else st.button(label=self.etiqueta, key=self.clave_widget)
        )
        if presionado:
            st.session_state[self.clave_estado] = True

    def fue_presionado(self) -> bool:
        """
        Verifica si el botón fue presionado en algún momento durante la sesión.

        Returns:
            bool: True si el botón fue presionado, False en caso contrario.
        """
        return st.session_state[self.clave_estado]

    def ocultar(self):
        """Oculta el botón limpiando el placeholder."""
        self._ph.empty()
        st.rerun()      
        
    def reiniciar(self) -> None:
        """
        Restablece el estado del botón a False.

        Esto permite que el botón pueda ser presionado y detectado nuevamente.
        """
        st.session_state[self.clave_estado] = False


class SelectorFechasEvento:
    """
    Clase encargada de manejar la selección, validación y cálculo del rango de fechas.

    Al confirmar, guarda automáticamente los valores en st.session_state.

    Attributes:
        fecha_inicio (date): Fecha seleccionada de inicio.
        fecha_fin (date): Fecha seleccionada de fin.
        dias (int): Número de días entre inicio y fin, ambos inclusive.
        mes (str): Mes capitalizado de la fecha de inicio.
        es_valido (bool): Si el rango de fechas es válido.
        confirmado (bool): Si el usuario confirmó el rango de fechas.
    """

    def __init__(self):
        self.fecha_inicio = datetime.today().date()
        self.fecha_fin = datetime.today().date()
        self.dias = 0
        self.mes = ""
        self.es_valido = False

    def mostrar_controles(self):
        """
        Despliega los controles de selección de fechas y calcula los valores si son válidos.
        """
        st.markdown("## Seleccione fecha de inicio y fin del evento")
        col1, col2 = st.columns(2)

        with col1:
            self.fecha_inicio = st.date_input(
                "📅 Fecha de inicio", value=self.fecha_inicio
            )

        with col2:
            self.fecha_fin = st.date_input("📅 Fecha de fin", value=self.fecha_fin)

        if self.fecha_fin < self.fecha_inicio:
            st.error("❌ La fecha de fin no puede ser anterior a la de inicio.")
            self.es_valido = False
            return

        self.dias = (self.fecha_fin - self.fecha_inicio).days + 1
        self.mes = self.fecha_inicio.strftime("%B").capitalize()
        self.es_valido = True

    def obtener_resultado(self):
        """
        Devuelve los datos de las fechas si han sido confirmadas.

        Returns:
            dict | None: Diccionario con fecha_inicio, fecha_fin, dias y mes si confirmado; None en caso contrario.
        """
        return {
            "fecha_inicio": self.fecha_inicio,
            "fecha_fin": self.fecha_fin,
            "mes": self.mes,
        }

    def _guardar_en_session_state(self):
        """
        Guarda los valores confirmados en st.session_state con claves estándar.
        """
        st.session_state["fecha_inicio_evento"] = self.fecha_inicio
        st.session_state["fecha_fin_evento"] = self.fecha_fin
        st.session_state["mes_evento"] = self.mes

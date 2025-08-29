from Utils.utils import procesar_configuracion
from loguru import logger


class ConfigLoader:
    """
    Clase encargada de cargar, combinar y exponer la configuración de la aplicación.
    
    Carga un archivo interno de configuración estructural y un archivo editable por el usuario.
    Si hay claves duplicadas, los valores definidos por el usuario tienen prioridad.
    """

    def __init__(
        self,
        config_file: str = "Controllers/settings/config.yml",
        editable_file: str = "editable.yml"
    ):
        self._config_interna = procesar_configuracion(config_file)
        self._config_usuario = self._cargar_config_usuario(editable_file)
        self._config = self._combinar_configuraciones()

    def _cargar_config_usuario(self, path: str) -> dict:
        try:
            return procesar_configuracion(path)
        except FileNotFoundError:
            logger.warning(f"No se encontró archivo de configuración editable en: {path}. Se usará solo la configuración base.")
            return {}

    def _combinar_configuraciones(self) -> dict:
        """
        Combina la configuración interna y editable.
        En caso de conflicto, los valores del usuario sobrescriben.

        Returns:
            dict: configuración unificada.
        """
        combinada = self._config_interna.copy()
        combinada.update(self._config_usuario)
        return combinada

    @property
    def config(self) -> dict:
        """Devuelve el diccionario combinado de configuración final."""
        return self._config

    def get_config(self, *claves: str, por_defecto=None):
        """
        Accede a una clave anidada de la configuración de forma segura.

        Args:
            *claves: Secuencia de claves anidadas, por ejemplo: ("insumos", "drivers", "nom_base").
            por_defecto: Valor a retornar si la clave no existe (por defecto: None).

        Returns:
            Cualquier valor de configuración encontrado o el valor por defecto.
        """
        actual = self._config
        for clave in claves:
            if isinstance(actual, dict) and clave in actual:
                actual = actual[clave]
            else:
                logger.debug(f"Clave de configuración no encontrada: {' > '.join(claves)}")
                return por_defecto
        return actual

class ConfigClaves:
    """
    Clase auxiliar que expone claves de configuración de uso frecuente como atributos.

    Esta clase actúa como un 'wrapper' sobre una instancia de `ConfigLoader`, permitiendo
    acceder a ciertas claves anidadas del archivo de configuración de forma directa, 
    evitando el uso repetitivo de `get_config(...)` en otras partes del código.

    Atributos públicos comunes:
        COL_NRO_CUENTA (str): Nombre de la columna de número de cuenta.
        COL_SOCIEDAD (str): Nombre de la columna de sociedad.
        COL_X (str): Nombre de la columna auxiliar derivada con 1 carácter.
        COL_Y (str): Nombre de la columna auxiliar derivada con 3 caracteres.
        COL_AGRUPADOR_BASE (str): Clave de agrupador base.
        COL_DESC_CUENTA (str): Clave para la descripción de cuenta.
        AGRUPADOR_MAP (dict): Diccionario para mapear agrupadores base.
        HOJAS_DRIVERS (list[str]): Lista de nombres de hoja para los drivers.
        REN_ACUMULADOS (dict): Diccionario para renombrar columnas en acumulados.
        REN_UTILIDAD (dict): Diccionario para renombrar columnas en utilidad.
        VALORES_SOCIEDAD (list[str]): Lista de valores válidos para sociedad.
        DICT_NUM_MESES (dict): Diccionario de equivalencia mes → número.
        DICT_CLAVES_MESES (dict): Diccionario de equivalencia clave mes → nombre mes.
    """

    def __init__(self, config_loader: ConfigLoader):
        """
        Inicializa la clase a partir de una instancia de ConfigLoader.

        Args:
            config_loader (ConfigLoader): Instancia ya cargada del cargador de configuración.
        """
        self.config_loader = config_loader
        self.config = config_loader.config
        self.get_config = config_loader.get_config
        self._cargar_claves()

    def _cargar_claves(self) -> None:
        """
        Carga en atributos internos las claves más utilizadas del archivo de configuración.

        Extrae valores anidados de secciones como "dict_cols", "insumos_drivers", "valores_filtrar", etc.,
        y los guarda como atributos accesibles directamente, reduciendo la repetición de llamadas.

        Este método se llama automáticamente al inicializar la clase.
        """
        # Claves principales.
        
        # Diccionario de columnas de dataframes.
        dict_cols = self.get_config("dict_cols")
        
        

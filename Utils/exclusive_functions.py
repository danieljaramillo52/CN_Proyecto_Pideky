# utils_filtros.py
import re
import datetime as dt
from typing import Dict, List, Tuple, Any, Optional, Sequence
import pandas as pd
import streamlit as st


class FiltroMultiple:
    """
    Gestiona la creación de filtros dinámicos sobre un DataFrame
    con integración en Streamlit. Admite filtros por columnas
    categóricas y un tratamiento especial de la columna 'Fecha'
    (filtro por rango inicio/fin).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        columnas_candidatas: List[str] | None = None,
        key_prefix: str = "flt",
    ):
        """
        Inicializa el filtro múltiple.

        Args:
            df: DataFrame base a filtrar.
            columnas_candidatas: Columnas a mostrar en el selector de filtros.
                                 Si es None, se muestran todas.

            key_prefix: Prefijo único para las claves en session_state.
        """
        self.df = df
        self.key = key_prefix

        self.columnas_candidatas = (
            list(columnas_candidatas) if columnas_candidatas else list(df.columns)
        )

        # Manejo especial de fechas
        self.tiene_fecha = "Fecha" in df.columns
        self.k_range = f"{self.key}_range_Fecha"
        self.fmin, self.fmax = (
            self._preparar_limites_fecha() if self.tiene_fecha else (None, None)
        )

    # ---------- Helpers ----------

    def _preparar_limites_fecha(self) -> Tuple[dt.date, dt.date]:
        """
        Obtiene los límites mínimo y máximo de la columna 'Fecha'.
        Si no hay valores válidos, devuelve un rango por defecto.
        """
        serie = pd.to_datetime(self.df["Fecha"], errors="coerce", dayfirst=False)
        if serie.notna().any():
            return serie.min().date(), serie.max().date()
        return dt.date.today() - dt.timedelta(days=30), dt.date.today()

    @staticmethod
    def _safe_name(col: str) -> str:
        """
        Devuelve un nombre seguro para usar como clave en session_state.
        """
        return re.sub(r"[^A-Za-z0-9_]+", "_", str(col))

    # ---------- UI ----------

    def _render_form(self) -> Tuple[bool, bool, List[str], Dict[str, List[str]]]:
        """
        Renderiza el formulario de filtros en Streamlit.

        Returns:
            aplicar: True si se pulsó "Aplicar filtros".
            limpiar: True si se pulsó "Limpiar filtros".
            cols_elegidas: Lista de columnas seleccionadas.
            filtros_categoricos: Diccionario {columna: valores} sin incluir 'Fecha'.
        """
        filtros_categoricos: Dict[str, List[str]] = {}

        with st.form(key=f"{self.key}_form"):
            st.write("### Columnas para filtrar")
            cols_elegidas = st.multiselect(
                "Selecciona las columnas que quieres filtrar",
                options=self.columnas_candidatas,
                key=f"{self.key}_cols",
            )

            for col in cols_elegidas:
                if col == "Fecha":
                    self._render_selector_fecha()
                    continue
                self._render_selector_categoria(col, filtros_categoricos)

            aplicar = st.form_submit_button("✅ Aplicar filtros")
            limpiar = st.form_submit_button("🧹 Limpiar filtros")

        return aplicar, limpiar, cols_elegidas, filtros_categoricos

    def _render_selector_fecha(self) -> None:
        """
        Renderiza un selector de rango de fechas.
        """
        if self.fmin and self.fmax:
            default_range = st.session_state.get(self.k_range, (self.fmin, self.fmax))
            if not (isinstance(default_range, (list, tuple)) and len(default_range) == 2):
                default_range = (self.fmin, self.fmax)
            st.date_input(
                "Rango de fechas",
                value=default_range,
                min_value=self.fmin,
                max_value=self.fmax,
                key=self.k_range,
            )
        else:
            st.warning("No se pudo interpretar la columna 'Fecha'.")

    def _render_selector_categoria(self, col: str, filtros: Dict[str, List[str]]) -> None:
        """
        Renderiza un multiselect para una columna categórica.

        Args:
            col: Nombre de la columna.
            filtros: Diccionario donde se almacenan los filtros elegidos.
        """
        safe = self._safe_name(col)
        k = f"{self.key}_val_{safe}"
        universo = self.df[col]
        opciones = sorted(pd.Series(universo).dropna().astype(str).unique())

        selec = st.multiselect(
            f"Filtrar por: {col}",
            options=opciones,
            default=st.session_state.get(k, []),
            key=k,
        )
        if selec:
            filtros[col] = selec

    # ---------- Acciones ----------

    def _aplicar(
        self, cols_elegidas: List[str], filtros_categ: Dict[str, List[str]]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Aplica los filtros categóricos y de fecha al DataFrame.

        Returns:
            df_filtrado: DataFrame con los filtros aplicados.
            filtros_aplicados: Diccionario de filtros usados.
        """
        df_filtrado = self.df.copy()
        filtros_aplicados: Dict[str, Any] = {}

        # Categóricos
        for col, valores in filtros_categ.items():
            if col == "Fecha":
                continue
            df_filtrado = df_filtrado[df_filtrado[col].astype(str).isin(valores)]
            filtros_aplicados[col] = valores

        # Fecha
        if "Fecha" in cols_elegidas and self.tiene_fecha:
            rango = st.session_state.get(self.k_range, (self.fmin, self.fmax))
            if isinstance(rango, (list, tuple)) and len(rango) == 2:
                fecha_ini, fecha_fin = rango
                ser = pd.to_datetime(df_filtrado["Fecha"], errors="coerce", dayfirst=False)
                df_filtrado = df_filtrado[
                    (ser >= pd.Timestamp(fecha_ini)) & (ser <= pd.Timestamp(fecha_fin))
                ]
                filtros_aplicados["Fecha"] = (fecha_ini, fecha_fin)

        st.success(f"Filas resultantes: {len(df_filtrado):,}")
        return df_filtrado, filtros_aplicados

    def _limpiar(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Limpia el estado de los filtros en session_state y devuelve
        el DataFrame original.
        """
        for col in self.columnas_candidatas:
            safe = self._safe_name(col)
            st.session_state.pop(f"{self.key}_val_{safe}", None)
        st.session_state.pop(self.k_range, None)
        st.session_state.pop(f"{self.key}_cols", None)

        st.info("Filtros limpiados.")
        st.rerun()
        return self.df, {}

    # ---------- Orquestador ----------

    def run(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Orquesta el flujo:
        1. Renderizar formulario
        2. Aplicar filtros si procede
        3. Limpiar si procede
        4. Retornar DataFrame filtrado y filtros aplicados
        """
        aplicar, limpiar, cols_elegidas, filtros_categ = self._render_form()

        if aplicar:
            return self._aplicar(cols_elegidas, filtros_categ)
        if limpiar:
            return self._limpiar()
        return self.df, {}


# --------- Orquestador con MISMA API ---------
def filtro_multiple(
    df: pd.DataFrame,
    columnas_candidatas: list[str] | None = None,
    key_prefix: str = "flt",
):
    """
    API de compatibilidad 1:1 con la función original.
    Retorna un DataFrame filtrado y un dict con filtros aplicados.
    """
    return FiltroMultiple(
        df=df,
        columnas_candidatas=columnas_candidatas,
        key_prefix=key_prefix,
    ).run()


FMT = "%Y.%m.%d"  # AAAA.MM.DD

def construir_filtros_desde_df_ymd_dot(
    df: pd.DataFrame,
    columnas: Optional[Sequence[str]] = None,
    col_fecha: str = "Fecha",
    max_unicos: int = 200,
) -> Dict[str, Any]:
    """
    Crea un dict filtros_plantilla a partir del DF.
    - Columnas normales -> lista de únicos (sin nulos).
    - col_fecha -> tupla (min, max) en formato AAAA.MM.DD.
    """
    if columnas is None:
        columnas = list(df.columns)

    filtros: Dict[str, Any] = {}
    for col in columnas:
        if col not in df.columns:
            continue

        if col == col_fecha:
            # Convierte asumiendo FMT fijo (coerce para evitar errores en valores malos)
            fechas = pd.to_datetime(df[col], format=FMT, errors="coerce").dropna()
            if fechas.empty:
                continue
            filtros[col] = (fechas.min().strftime(FMT), fechas.max().strftime(FMT))
        else:
            s = df[col].dropna()
            if s.empty:
                continue
            unicos = pd.unique(s)
            if len(unicos) == 0 or len(unicos) > max_unicos:
                continue
            try:
                unicos = sorted(unicos.tolist())
            except Exception:
                unicos = list(unicos)
            filtros[col] = unicos

    return filtros


def aplicar_filtros_plantilla_ymd_dot(
    df: pd.DataFrame,
    filtros: Dict[str, Any],
    col_fecha: str = "Fecha",
) -> pd.DataFrame:
    """
    Aplica filtros; para col_fecha usa rango (inicio, fin) en AAAA.MM.DD.
    Cualquier valor de fecha inválido se descarta silenciosamente.
    """
    out = df.copy()
    for col, v in filtros.items():
        if col not in out.columns:
            continue

        if col == col_fecha and isinstance(v, tuple) and len(v) == 2:
            ini = pd.to_datetime(v[0], format=FMT, errors="coerce")
            fin = pd.to_datetime(v[1], format=FMT, errors="coerce")
            if pd.isna(ini) or pd.isna(fin):
                continue

            # Parse columna con el formato fijo AAAA.MM.DD
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                fechas = pd.to_datetime(out[col], errors="coerce")
            else:
                fechas = pd.to_datetime(out[col], format=FMT, errors="coerce")

            mask = fechas.between(ini, fin, inclusive="both") & fechas.notna()
            out = out[mask]
        else:
            # Inclusión normal por valores
            vals = v if isinstance(v, (list, tuple, set)) else [v]
            if len(vals) == 0:
                continue
            out = out[out[col].isin(vals)]

    return out
from typing import Dict, Any
import os
from pathlib import Path
import pandas as pd
import streamlit as st
import Utils.ui_components as ui_comp
import Utils.utils as utils


class ProcesarInsumos:
    def __init__(self, config_insumos: Dict[str, Any], config_msg: Dict[str, Any],config_cols : Dict[str, Any] ):
        """
        Inicializa el controlador de insumos.

        Args:
            config_insumos: Configuración de rutas y archivos (dict).
            config_msg: Configuración de mensajes de UI (dict).
        """
        self.config_insumos = config_insumos
        self.cnf_msg = config_msg
        self.cnf_cols = config_cols


    def ejecutar_proceso_insumos(self) -> None:
        """
        Orquestador público:
        - Asegura que la clave 'etapa_insumos' exista en session_state.
        - Ejecuta la carga de insumos.
        - Muestra botón de confirmación sólo en la etapa inicial 'carga'.
        """
        st.session_state.setdefault("etapa_insumos", "carga")
        self._orquestar_cargue_insumos()

    def _orquestar_cargue_insumos(self) -> None:
        """
        Flujo de carga de insumos:
          1) Cargar ventas (si no existen en sesión).
          2) Cargar clientes (si no existen en sesión).
          3) Renderizar botón de confirmación SOLO si la etapa es 'carga'.
        """
        msg_vtas = st.empty()
        msg_clientes = st.empty()
        msg_plantilla = st.empty()
        
        self._cargar_ventas_si_necesario(msg_vtas)
        self._cargar_base_clientes(msg_clientes)
        self._cargar_plantilla_filtros(msg_plantilla)
        
        if st.session_state.get("etapa_insumos") != "carga":
            return  # No mostrar botón de nuevo

        cfg_btn = self.config_insumos["cnf_botones"]["confirmar_insumos"]
        btn_confirmar = ui_comp.ButtonTracker(**cfg_btn)

        if btn_confirmar.fue_presionado():
            msg_vtas.empty()
            msg_clientes.empty()
            msg_plantilla.empty()
            
            st.session_state["etapa_insumos"] = "listo"
            btn_confirmar.reiniciar()
            st.rerun()

    # ---------- Carga de insumos ----------

    def _cargar_ventas_si_necesario(self, msg) -> None:
        """
        Carga los archivos de ventas solo si aún no están en sesión.

        Args:
            msg: contenedor Streamlit para mostrar mensajes.
        """
        if "archivos_insumos" in st.session_state:
            return

        msg.info(self.cnf_msg["msg_vtas_procesando"])
        dict_archivos_vtas = {}
        fallidos = []

        dir_vtas = self.config_insumos["path_insumos_vtas"]
        if not os.path.isdir(dir_vtas):
            msg.error(f"Directorio de ventas no existe: {dir_vtas}")
            return

        rutas_arc_vtas = [
            os.path.join(dir_vtas, f)
            for f in os.listdir(dir_vtas)
            if f.lower().endswith(".csv")
        ]

        for path_arch in rutas_arc_vtas:
            try:
                dict_archivos_vtas[path_arch] = utils.load_csv(path=path_arch)
            except Exception as e:
                fallidos.append((str(path_arch), str(e)))

        ui_comp.set_key_ss_st("archivos_insumos", dict_archivos_vtas)

        if fallidos:
            msg.warning(
                f"Ventas cargadas: {len(dict_archivos_vtas)} | fallidos: {len(fallidos)}"
            )
        msg.success(self.cnf_msg["msg_vtas_procesadas"])

    def _cargar_base_clientes(self, msg) -> None:
        """
        Carga la base de clientes solo si aún no está en sesión.

        Args:
            msg: contenedor Streamlit para mostrar mensajes.
        """
        if "base_clientes" in st.session_state:
            return

        dir_clientes = self.config_insumos["path_insumos_clientes"]
        
        if not os.path.isdir(dir_clientes):
            msg.error(f"Directorio de clientes no existe: {dir_clientes}")
            return

        rutas_arc_clientes = [
            os.path.join(dir_clientes, f)
            for f in os.listdir(dir_clientes)
            if f.lower().endswith(".csv")
        ]
        if not rutas_arc_clientes:
            msg.error("No se encontraron CSV de clientes.")
            return

        msg.info(self.cnf_msg["msg_base_clientes_procesando"])
        base_clientes = utils.load_csv(rutas_arc_clientes[0])
        ui_comp.set_key_ss_st("base_clientes", base_clientes)
        msg.success(self.cnf_msg["msg_base_clientes_procesada"])

    def _cargar_plantilla_filtros(self, msg):
        """Proceso que carga una plantilla de filtros masivos definidos por el usuario.

        Carga la base de clientes solo si aún no está en sesión.

        Args:
            msg: contenedor Streamlit para mostrar mensajes.
        """
        
        if "plantilla_filtros" in st.session_state:
            return

        cnf_plantilla = self.config_insumos["plantilla_filtros"]
        dir_plantilla = cnf_plantilla["path_plantilla"]
        
        
        if not os.path.isdir(dir_plantilla):
            msg.error(f"Plantilla de filtros no exise {dir_plantilla}")
            return
        
        msg.info(self.cnf_msg["msg_base_plantilla_filtros_procesada"])
        
        plant_filtros = utils.lectura_simple_excel(dir_insumo=dir_plantilla, nom_insumo=cnf_plantilla["nom_base"], nom_hoja=cnf_plantilla["nom_hoja"])
        
        ui_comp.set_key_ss_st("plantilla_fitros", plant_filtros)
        
        msg.success(self.cnf_msg["msg_base_plantilla_filtros_procesada"])
        msg.empty()
        
    def _procesar_insumos_vtas(self, dict_archivos, df_clientes):
        """
        Consolida insumos de ventas y clientes con base en la configuración
        de columnas definida en `self.cnf_cols`.
        """
        cols_cfg = self.cnf_cols  # alias corto

        # Ordenar columnas
        dict_archivos_sorted = {
            key: df[sorted(df.columns)] for key, df in dict_archivos.items()
        }

        # Renombrar columnas
        dict_arch_ren_cols = {
            key: utils.renombrar_columnas_con_diccionario(
                df=df, cols_to_rename=self.config_insumos["dict_rename_df_vtas"]
            )
            for key, df in dict_archivos_sorted.items()
        }

        # Concatenar ventas
        df_vtas_completo = utils.concatenar_dataframes(
            df_list=list(dict_arch_ren_cols.values())
        )

        # Dividir ventas por modelo
        df_vtas_dir = utils.filtrar_por_valores(
            df_vtas_completo,
            columna=cols_cfg["ventas"]["col_modelo"],
            valores=cols_cfg["ventas"]["valor_directa"],
        )
        df_vtas_indir = utils.filtrar_por_valores(
            df_vtas_completo,
            columna=cols_cfg["ventas"]["col_modelo"],
            valores=cols_cfg["ventas"]["valor_directa"],
            incluir=False,
        )

        # Dividir clientes por modelo
        df_clientes_dir = utils.filtrar_por_valores(
            df_clientes,
            columna=cols_cfg["clientes"]["col_modelo"],
            valores=cols_cfg["ventas"]["valor_directa"],
        )
        df_clientes_indir = utils.filtrar_por_valores(
            df_clientes,
            columna=cols_cfg["clientes"]["col_modelo"],
            valores=cols_cfg["ventas"]["valor_directa"],
            incluir=False,
        )

        # Cruces
        df_vtas_indir_group = utils.pd_left_merge_two_keys(
            base_left=df_vtas_indir,
            base_right=df_clientes_indir[cols_cfg["clientes"]["cols_indirecta"]],
            right_key=["COD_AC", "COD_CRM"],
            left_key=[cols_cfg["ventas"]["col_jefe_ac"], cols_cfg["ventas"]["col_cliente"]],
        )

        df_vtas_dir_group = utils.pd_left_merge_two_keys(
            base_left=df_vtas_dir,
            base_right=df_clientes_dir[cols_cfg["clientes"]["cols_directa"]],
            left_key=cols_cfg["ventas"]["col_cliente"],
            right_key="CODIGO_ERP",
            conservar_llave_derecha=True,
        )

        # Reconstrucción
        df_vtas_completo = utils.concatenar_dataframes(
            [df_vtas_indir_group, df_vtas_dir_group]
        )

        # Numéricos
        for c in cols_cfg["ventas"]["claves_numericas"]:
            if c in df_vtas_completo.columns:
                df_vtas_completo[c] = df_vtas_completo[c].astype(float).round(2)

        ui_comp.set_key_ss_st("df_vtas_completo", df_vtas_completo)

        # Clientes nulos
        if df_vtas_completo is not None:
            df_vtas_ccup_null = df_vtas_completo[df_vtas_completo["CCUP"].isnull()][cols_cfg["cols_archivo_nulos"]]
            
            df_vtas_ccup_null_wo_duplicates = df_vtas_ccup_null.drop_duplicates()
            
            df_vtas_ccup_null_wo_duplicates.to_excel(cols_cfg["nombre_archivo_salida"], index=False)
            
            
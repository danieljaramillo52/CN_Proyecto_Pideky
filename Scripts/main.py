import config_path_routes
import pandas as pd
import streamlit as st
import Utils.utils as utils
from Controllers.procesar_insumos import ProcesarInsumos
from Controllers.config_loader import ConfigLoader, ConfigClaves
from Utils.ui_components import add_key_ss_st
from Utils.exclusive_functions import (
    filtro_multiple,
    aplicar_filtros_plantilla_ymd_dot,
    construir_filtros_desde_df_ymd_dot,
)

class Aplicacion:
    """Orquesta el flujo principal con etapas: 'carga' → 'listo' → 'filtrar'."""

    def __init__(self) -> None:
        self.config_loader = ConfigLoader()
        self.get_config = self.config_loader.get_config
        self.claves = ConfigClaves(self.config_loader)

        self.procesar_insumos = ProcesarInsumos(
            config_insumos=self.get_config("config_insumos"),
            config_msg=self.get_config("cnf_mensajes"),
            config_cols=self.get_config("config_cols"),
        )

    def ejecutar(self) -> None:
        self._inicializar_session()

        # 1) Cargue/orquestación (solo pinta botón en 'carga')
        self.procesar_insumos.ejecutar_proceso_insumos()

        # 2) Lee etapa y datos base
        etapa = st.session_state.get("etapa_insumos", "carga")
        dict_archivos = st.session_state.get("archivos_insumos", {})
        df_clientes = st.session_state.get("base_clientes", pd.DataFrame())

        # 3) Navegación por etapas
        if etapa == "listo":
            ph_btn = st.empty()  # placeholder para poder ocultarlo de inmediato

            if ph_btn.button("Procesar Formato", key="btn_procesar_formato", type="primary"):
                ph_btn.empty()

                with st.spinner(self.get_config("cnf_mensajes", "msg_procesando_insm")):
                    if "df_vtas_completo" not in st.session_state:
                        self.procesar_insumos._procesar_insumos_vtas(
                            dict_archivos=dict_archivos,
                            df_clientes=df_clientes,
                        )

                # Mensaje temporal para la siguiente pantalla
                st.session_state["flash_msg"] = self.get_config("cnf_mensajes", "msg_formato_procesado")
                st.session_state["etapa_insumos"] = "filtrar"
                st.rerun()

        elif etapa == "filtrar":
            # ---- Estado persistente de vista y resultados ----
            st.session_state.setdefault("vista_filtros", None)            # None | "manual" | "plantilla"
            st.session_state.setdefault("nonce_filtros", 0)               # para keys de widgets
            st.session_state.setdefault("df_agrupado_preview", pd.DataFrame())

            # Handlers de navegación (no cambian funcionalidad, solo controlan la vista)
            def _ir_manual():
                st.session_state["vista_filtros"] = "manual"

            def _ir_plantilla():
                st.session_state["vista_filtros"] = "plantilla"

            # Botones de modo (no se vacían al siguiente rerun)
            col1, col2 = st.columns(2)
            col1.button("Filtros manuales", key="btn_filtros_manuales", type="primary", on_click=_ir_manual)
            col2.button("Usar plantilla filtros", key="btn_usar_plantilla", on_click=_ir_plantilla)

            # ------------------- Vista: Manual -------------------
            if st.session_state["vista_filtros"] == "manual":
                df_base = st.session_state.get("df_vtas_completo")

                if df_base is None or getattr(df_base, "empty", True):
                    st.warning(self.get_config("cnf_mensajes", "msg_df_para_filtrar"))
                    if st.button("Volver", key="btn_volver_listo"):
                        st.session_state["etapa_insumos"] = "listo"
                        st.session_state["vista_filtros"] = None
                        st.rerun()
                else:
                    st.markdown(f"### {self.get_config('cnf_mensajes', 'msg_filtros_dinámicos')}")
                    df_filtrado, filtros = filtro_multiple(
                        df=df_base,
                        key_prefix=f"flt_vtas_{st.session_state['nonce_filtros']}",
                    )

                    # Mantén tu misma lógica de agrupación
                    df_agrupado = st.session_state.get("df_agrupado", pd.DataFrame())
                    df_agrupado = utils.group_by_and_operate(
                        df_filtrado,
                        group_col=self.get_config("columnas_finales_agrupación", "group"),
                        operation="sum",
                        operation_cols=self.get_config("columnas_finales_agrupación", "sum"),
                    )

                    # Persiste el resultado para que no “desaparezca”
                    st.session_state["df_agrupado"] = df_agrupado
                    st.session_state["df_agrupado_preview"] = df_agrupado

            # ------------------- Vista: Plantilla -------------------
            elif st.session_state["vista_filtros"] == "plantilla":
                msg_plantilla = st.empty()
                msg_plantilla.info("🔄 Aplicando filtros por medio de plantilla...")

                df_base = st.session_state.get("df_vtas_completo")
                df_plantilla = st.session_state.get("plantilla_fitros", {})  # se respeta tu clave

                dict_filtros = construir_filtros_desde_df_ymd_dot(df=df_plantilla)
                df_filtrado = aplicar_filtros_plantilla_ymd_dot(df=df_base, filtros=dict_filtros)

                df_agrupado = utils.group_by_and_operate(
                    df_filtrado,
                    group_col=self.get_config("columnas_finales_agrupación", "group"),
                    operation="sum",
                    operation_cols=self.get_config("columnas_finales_agrupación", "sum"),
                )

                msg_plantilla.success("Filtros aplicados por medio de plantilla ✅")

                # Persiste el resultado
                st.session_state["df_agrupado"] = df_agrupado
                st.session_state["df_agrupado_preview"] = df_agrupado

            # ------------------- Render común de preview + descarga -------------------
            df_prev = st.session_state.get("df_agrupado_preview", pd.DataFrame())
            if isinstance(df_prev, pd.DataFrame) and not df_prev.empty:
                st.markdown(f"### {self.get_config('cnf_mensajes', 'msg_vista_previa_resultado')}")
                st.dataframe(df_prev.head(30), use_container_width=True)

                csv_bytes = df_prev.to_csv(index=False).encode("utf-8-sig")
                cfg_btn_dw = self.get_config("config_insumos", "cnf_botones", "descargar_csv")
                st.download_button(
                    label=cfg_btn_dw["etiqueta"],
                    key=cfg_btn_dw["clave"],
                    data=csv_bytes,
                    file_name=cfg_btn_dw.get("file_name"),
                    mime=cfg_btn_dw.get("mime"),
                )

    def _inicializar_session(self) -> None:
        """Inicializa solo si no existen, para no pisar valores en cada rerun."""
        add_key_ss_st("etapa_insumos", "carga")
        add_key_ss_st("df_agrupado", pd.DataFrame())
        add_key_ss_st("vista_filtros", None)
        add_key_ss_st("nonce_filtros", 0)
        add_key_ss_st("df_agrupado_preview", pd.DataFrame())

if __name__ == "__main__":
    utils.setup_ui()
    app = Aplicacion()
    app.ejecutar()

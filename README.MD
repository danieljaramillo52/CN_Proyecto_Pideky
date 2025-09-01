# 📦 CN_Proyecto_Pideky – Paquete de Documentación

Este documento reúne, listo para copiar/pegar como archivos del repo, una documentación estándar basada en tu patrón **config.yml** / **editable.yml**, tu estructura de carpetas (Controllers, Scripts, Utils, Insumos, Resultados, static, etc.) y tus `.bat` de ejecución.

> Sugerencia de ubicación: crea una carpeta `docs/` para las guías y deja `README.md`, `CHANGELOG.md` y `CONTRIBUTING.md` en la raíz.

---

## 1) `README.md`

```markdown
# CN_Proyecto_Pideky

Automatización de ETL y postprocesos para informes de ventas Pideky (Comercial Nutresa), usando Python y configuración por YAML. El flujo toma **Insumos** (archivos Excel/CSV), aplica transformaciones y reglas, y genera **Resultados** estandarizados (Descargable desde la interfaz de streamlit.)

## 🚀 Características
- Configuración externa (YAML): separación **config.yml** (núcleo) y **editable.yml** (ajustes del usuario).
- Procesamiento por módulos (`Controllers/`, `Utils/`, `Scripts/`).
- Ejecutables por `.bat` para Windows (entorno embebido o venv).
- Salidas reproducibles en `Resultados/`.

## 📁 Estructura
```
| Ruta/Elemento           | Tipo    | Descripción breve                                    |
| ----------------------- | ------- | ---------------------------------------------------- |
| `Controllers/`          | Carpeta | Orquestación y lógica por pasos                      |
| `Scripts/`              | Carpeta | Puntos de entrada y automatizaciones (CLI/Batch)     |
| `Utils/`                | Carpeta | Funciones reutilizables (I/O, validaciones, helpers) |
| `Insumos/`              | Carpeta | Archivos de entrada *(no versionar)*                 |
| `Resultados/`           | Carpeta | Archivos de salida *(no versionar)*                  |
| `static/`               | Carpeta | Recursos estáticos (logos, CSS, etc.)                |
| `Img/`                  | Carpeta | Imágenes/diagramas                                   |
| `.gitignore`            | Archivo | Reglas de exclusión de versionado                    |
| `requirements.txt`      | Archivo | Dependencias de Python                               |
| `editable.yml`          | Archivo | Configuración editable por el usuario                |
| `iniciar.bat`           | Archivo | Preparación del entorno e instalación                |
| `ejecutar.bat`          | Archivo | Ejecución del flujo principal                        |
| `conversor_insumos.bat` | Archivo | Automatización para conversión de insumos            |
```

## 🔧 Requisitos
- Windows 10/11 o Linux/macOS (opcional)
- Python 3.10–3.12 (o **entorno embebido** provisto)

## 🛠️ Instalación
**Opción A – Entorno embebido (recomendado en Windows):**
1. Descomprime `python-3.12.5-emb.zip` en la raíz del proyecto.
2. Ejecuta `iniciar.bat` (crea/activa entorno y dependencias) y luego `ejecutar.bat`.

**Opción B – Virtualenv:**
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## ⚙️ Configuración por YAML
- `config.yml`: rutas y constantes **internas** (no tocar en uso normal).
- `editable.yml`: parámetros **de usuario** (columnas ajustables). 
Las rutas de insumos se obtienen automaticamente de manera recursiva. Solo basta con generar los archivos utilizando **conversor_insumos.bat**
> Ver `docs/CONFIGURACION.md` para el detalle de claves y ejemplos.

## ▶️ Ejecución
- **Batch**: doble clic en `ejecutar.bat`.
- **CLI** (ejemplo genérico):
```bat
.venv\Scripts\activate
python Scripts/main.py --modo produce --periodo 2025-01
```
- **Streamlit** (si aplica):
```bat
.venv\Scripts\activate
streamlit run Scripts/app.py
```


##  Licencia
MIT License

Copyright (c) 2025 Daniel Jaramillo Bustamante - XpertGroup.sas

daniel.jaramillo@xpertgroup.co


---

## 2) `docs/CONFIGURACION.md`

```markdown
# Guía de configuración (editable.yml)

Este proyecto separa configuración **interna** config.yml de **editable** (usuario).

## `config.yml` (interno/protegido)
**Objetivo:** rutas base, nombres “oficiales”, columnas canónicas, y banderas de ejecución por defecto.
```

## `editable.yml` (usuario)
**Objetivo:** seleccionar formato final de resultados. (Columnas para agrupar y sumar en la base filtrada final.)

### Buenas prácticas
- Mantén `config.yml` **inmutable** 
- Deja `editable.yml` con **valores por defecto** 

## 3) `docs/USO.md`

## Guía de uso
## 1) Preparar Insumos
1. Utiliza el archivo **conversor_insumos.bat** para generar insumos necesarios para la automatización. 

2. Verifica pestañas/hojas esperadas y tipos de datos (fechas, numéricos).

## 2) Configurar
- Edita `editable.yml` con las columnas que quieras agrupar sigueindo la lista de columnas propuestas. 


---
##  ejemplo  contenido editable.yml:  

```yaml
# Ejemplo 1. Tiempo x Canal x Marca
columnas_finales_agrupación:
  group:
    - Tiempo x Canal x Marca
    - "Canal"
    - "Sub Canal"
    - "Marca"
  sum:
    - "Venta act $"
    - "Venta act Und"
    - "Venta act kg"
```
---

```yaml
# Ejemplo 2 – Cliente x Región x Oficina
columnas_finales_agrupación:
  group:
    - "Cod Cliente"
    - "Nom Cliente"
    - "Región"
    - "Oficina de ventas"
    - "Municipio"
  sum:
    - "Venta act $"
    - "Venta act Und"
    - "Venta act kg"
```
---
```yaml
# Ejemplo 3 –Ejemplo 3 – Producto x Categoría x Atención 

# Suma unicamente por $ y Und.

columnas_finales_agrupación:
  group:
    - "Cod Material"
    - "Desc Material"
    - "Categoría"
    - "Sub Categoría"
    - "Línea"
    - "Modelo de atención"
  sum:
    - "Venta act $"
    - "Venta act Und"
```


## 3) Ejecutar
- **Windows (embebido/venv)**: usa `ejecutar.bat`.
- **Línea de comando** (ejemplo genérico):
```bat
  python Scripts/main.py --periodo 2025-01 --exportar_csv
```

## 4) Resultados
- Archivos consolidados en `Resultados/` clientes_sin_ccup.xlsx 
Archivo que contiene los códigos que no cruzaron inicialmente. 

--- 

## Capas
- **Utils**: lectura/escritura (Excel, CSV), normalización, validaciones, tipado.
- **Controllers**: procesa configuración del proyecto ordena pasos del pipeline (lectura → transform → exportación).
- **Scripts**: puntos de entrada (CLI/GUI/Batch).

## Decisiones (ADR breve)
- **YAML externo**: separar `config.yml` (protección) vs `editable.yml` (usuario) para trazabilidad.
- **Estructura Insumos/Resultados**: no versionar archivos pesados → reproducibilidad y repos livianos.
```

---

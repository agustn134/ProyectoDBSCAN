"""
utils/data_handler.py
---------------------
Toda la logica de carga, limpieza, mapeo y filtrado de datos.
No contiene nada de Streamlit excepto lo estrictamente necesario para el retorno.
"""

import pandas as pd
import numpy as np
import unicodedata


# Columnas esperadas del CSV de KoboToolbox
COLUMNAS_CLAVE = [
    "Edad", "Genero",
    "P1_Destino", "P2_Estres", "P3_Frecuencia",
    "P4_NivelEstres", "P5_FinSemana",
]

# Columnas numericas derivadas del mapeo A/B/C/D
FEATURES = [
    "P1_Destino_num", "P2_Estres_num", "P3_Frecuencia_num",
    "P4_NivelEstres", "P5_FinSemana_num",
]

# Mapas de interpretacion para el reporte automatico
MAPA_P1 = {
    1: "Ahorro/inversion",
    2: "Entretenimiento",
    3: "Necesidades basicas",
    4: "Experiencias/viajes",
}
MAPA_P2 = {
    1: "Actividad fisica",
    2: "Salidas sociales",
    3: "Descanso en casa",
    4: "Consumo digital",
}


def _extraer_letra(texto: str) -> float:
    """Convierte respuesta tipo 'A) texto' en 1, 2, 3 o 4. Devuelve NaN si no reconoce."""
    texto = str(texto)
    if "A)" in texto: return 1.0
    if "B)" in texto: return 2.0
    if "C)" in texto: return 3.0
    if "D)" in texto: return 4.0
    return np.nan


def _limpiar_texto(s: str) -> str:
    """Normaliza texto: pasa a minúsculas, quita espacios y elimina acentos/marcas diacríticas."""
    s = str(s).lower().strip()
    s = "".join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s


def _obtener_valores_coalescidos(df_raw: pd.DataFrame, keywords: list, exclude: list = None) -> pd.Series:
    """
    Busca todas las columnas que contengan alguna de las palabras clave (y no contengan las de exclusión)
    y las combina de izquierda a derecha (coalesce/combine_first) para no perder datos entre versiones de formulario.
    """
    keywords_clean = [_limpiar_texto(kw) for kw in keywords]
    exclude_clean = [_limpiar_texto(ex) for ex in exclude] if exclude else []
    
    cols_to_combine = []
    for col in df_raw.columns:
        col_clean = _limpiar_texto(col)
        # Verificar exclusiones
        if any(ex in col_clean for ex in exclude_clean):
            continue
        # Verificar inclusiones
        if any(kw in col_clean for kw in keywords_clean):
            cols_to_combine.append(col)
            
    if not cols_to_combine:
        return pd.Series([np.nan] * len(df_raw))
        
    res = df_raw[cols_to_combine[0]]
    for col in cols_to_combine[1:]:
        res = res.combine_first(df_raw[col])
    return res


def cargar_y_limpiar(uploaded_file) -> pd.DataFrame:
    """
    Lee el CSV subido por el usuario, mapea dinámicamente las columnas basándose en palabras clave
    para soportar distintas versiones del formulario de KoboToolbox, limpia y convierte valores vacíos.

    Returns
    -------
    pd.DataFrame con columnas originales + columnas _num derivadas.
    """
    df_raw = pd.read_csv(uploaded_file, sep=";")

    df = pd.DataFrame(index=df_raw.index)

    columnas_clave = ['Edad', 'Genero', 'P1_Destino', 'P2_Estres', 'P3_Frecuencia', 'P4_NivelEstres', 'P5_FinSemana']
    
    # Si el CSV ya viene limpio (ej. datos_5000.csv), usamos las columnas directamente
    if all(col in df_raw.columns for col in columnas_clave):
        df = df_raw[columnas_clave].copy()
    else:
        # Bloque unificado (basado en el de generar_datos.py) para mapear columnas viejas y nuevas del form real
        try:
            df['Edad'] = df_raw.iloc[:, 0].fillna(df_raw.iloc[:, 9])
            df['Genero'] = df_raw.iloc[:, 1].fillna(df_raw.iloc[:, 9])
            df['P1_Destino'] = df_raw.iloc[:, 2].fillna(df_raw.iloc[:, 11])
            df['P2_Estres'] = df_raw.iloc[:, 3].fillna(df_raw.iloc[:, 17])
            df['P3_Frecuencia'] = df_raw.iloc[:, 4].fillna(df_raw.iloc[:, 19])
            df['P4_NivelEstres'] = df_raw.iloc[:, 6].fillna(df_raw.iloc[:, 20])
            df['P5_FinSemana'] = df_raw.iloc[:, 5].fillna(df_raw.iloc[:, 21])
        except IndexError:
            # Por seguridad si el archivo tiene menos columnas, tomamos las primeras 7 en el orden correcto
            for i, col in enumerate(columnas_clave):
                if i < len(df_raw.columns):
                    df[col] = df_raw.iloc[:, i]

    # Conversión y limpieza de tipo de datos
    df["Edad"]              = pd.to_numeric(df["Edad"], errors="coerce")
    df["P1_Destino_num"]    = df["P1_Destino"].apply(_extraer_letra)
    df["P2_Estres_num"]     = df["P2_Estres"].apply(_extraer_letra)
    df["P3_Frecuencia_num"] = df["P3_Frecuencia"].apply(_extraer_letra)
    df["P4_NivelEstres"]    = pd.to_numeric(df["P4_NivelEstres"], errors="coerce")
    df["P5_FinSemana_num"]  = df["P5_FinSemana"].apply(_extraer_letra)

    # Eliminar filas con datos faltantes en Edad o en las características del modelo
    df = df.dropna(subset=["Edad"] + FEATURES)
    df["Edad"] = df["Edad"].astype(int)
    return df


def aplicar_filtros(df: pd.DataFrame, genero: str, rango_edad: tuple) -> pd.DataFrame:
    """Aplica filtros de genero y rango de edad al DataFrame limpio."""
    df_f = df.copy()
    if genero != "Todos":
        df_f = df_f[df_f["Genero"] == genero]
    df_f = df_f[
        (df_f["Edad"] >= rango_edad[0]) & (df_f["Edad"] <= rango_edad[1])
    ]
    return df_f


def resumen_estadistico(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve describe() de las columnas numericas, redondeado a 2 decimales."""
    return df[FEATURES].describe().round(2)

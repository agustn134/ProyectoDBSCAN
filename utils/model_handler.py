"""
utils/model_handler.py
----------------------
Logica de entrenamiento (DBSCAN / K-Means), PCA, guardado de modelo
y generacion de metadatos y reporte de texto.
No contiene nada de Streamlit.
"""

import json
import datetime
import joblib

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA

from utils.data_handler import FEATURES, MAPA_P1, MAPA_P2

# Pesos teoricos aplicados post-estandarizacion
WEIGHTS = np.array([0.35, 0.25, 0.20, 0.10, 0.10])

PESOS_INFO = [
    ("P1 — Destino del gasto",       35, "Determina la asignación principal del presupuesto."),
    ("P2 — Manejo del estrés",        25, "Identifica hábitos de consumo compensatorio o preventivo."),
    ("P3 — Frecuencia de salidas",    20, "Mide la intensidad de consumo en actividades sociales."),
    ("P4 — Nivel de estrés (1-10)",   10, "Aporta precisión numérica al estado de ánimo."),
    ("P5 — Actividad fin de semana",  10, "Contextualiza la rutina de consumo temporal."),
]

COLORES_PALETTE = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444", "#06b6d4"]


def entrenar(
    df: pd.DataFrame,
    algoritmo: str,
    eps: float = 0.5,
    min_samples: int = 3,
    k_clusters: int = 3,
) -> dict:
    """
    Entrena el algoritmo seleccionado y devuelve un dict con todos los resultados.

    Parameters
    ----------
    df          : DataFrame filtrado (debe tener las columnas de FEATURES)
    algoritmo   : "DBSCAN (recomendado)" o "K-Means"
    eps, min_samples : parametros de DBSCAN
    k_clusters  : parametro de K-Means

    Returns
    -------
    dict con claves:
        nombre_alg, param_display, labels, df_resultado,
        X_pca, pca, varianza, modelo_obj, metadatos
    """
    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_weighted = X_scaled * WEIGHTS

    if "DBSCAN" in algoritmo:
        modelo_fit    = DBSCAN(eps=eps, min_samples=min_samples)
        labels        = modelo_fit.fit_predict(X_weighted)
        nombre_alg    = "DBSCAN"
        param_display = f"eps={eps}, min_samples={min_samples}"
    else:
        modelo_fit    = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        labels        = modelo_fit.fit_predict(X_weighted)
        nombre_alg    = "K-Means"
        param_display = f"k={k_clusters}"

    pca       = PCA(n_components=2)
    X_pca     = pca.fit_transform(X_weighted)
    varianza  = pca.explained_variance_ratio_

    df_resultado             = df.copy()
    df_resultado["Cluster"]  = labels
    df_resultado["PCA_1"]    = X_pca[:, 0]
    df_resultado["PCA_2"]    = X_pca[:, 1]

    modelo_obj = {
        "algoritmo": nombre_alg,
        "modelo":    modelo_fit,
        "pca":       pca,
        "scaler":    scaler,
        "weights":   WEIGHTS,
    }

    metadatos = {
        "fecha_entrenamiento":   datetime.datetime.now().isoformat(timespec="seconds"),
        "algoritmo":             nombre_alg,
        "parametros":            param_display,
        "n_registros":           int(len(df_resultado)),
        "ponderacion": {
            "P1_Destino":     0.35,
            "P2_Estres":      0.25,
            "P3_Frecuencia":  0.20,
            "P4_NivelEstres": 0.10,
            "P5_FinSemana":   0.10,
        },
        "clusters_encontrados": int(len([l for l in set(labels) if l != -1])),
        "puntos_ruido":         int(list(labels).count(-1)),
    }

    return {
        "nombre_alg":    nombre_alg,
        "param_display": param_display,
        "labels":        labels,
        "df_resultado":  df_resultado,
        "X_pca":         X_pca,
        "pca":           pca,
        "varianza":      varianza,
        "modelo_obj":    modelo_obj,
        "metadatos":     metadatos,
    }


def calcular_epsilon_sugerido(df: pd.DataFrame, min_samples: int = 6) -> float:
    """
    Calcula el valor óptimo de epsilon con auto-afinación.

    1. Calcula las k-distancias ordenadas.
    2. Intenta el método del codo (máx. segunda derivada).
    3. Valida que el eps resultante produzca entre 3-6 clusters.
    4. Si no lo logra, hace búsqueda binaria sobre el rango de distancias
       para encontrar el eps que genera ~4 clusters.

    Parameters
    ----------
    df          : DataFrame limpio con las columnas FEATURES
    min_samples : valor de min_samples que se usará

    Returns
    -------
    float con el valor de epsilon sugerido, redondeado a 2 decimales
    """
    from sklearn.neighbors import NearestNeighbors

    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_weighted = X_scaled * WEIGHTS

    k = max(min_samples - 1, 1)
    k = min(k, len(X_weighted) - 1)

    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X_weighted)
    distances, _ = neigh.kneighbors(X_weighted)

    distancias_ordenadas = np.sort(distances[:, k - 1], axis=0)

    # --- Paso 1: método del codo (segunda derivada) ---
    if len(distancias_ordenadas) > 2:
        segunda_derivada = np.diff(distancias_ordenadas, n=2)
        idx_codo = np.argmax(segunda_derivada) + 2
        eps_codo = float(distancias_ordenadas[idx_codo])
    else:
        eps_codo = float(np.median(distancias_ordenadas))

    # --- Paso 2: validar con DBSCAN rápido ---
    def contar_clusters(eps_val):
        """Retorna cuántos clusters (sin contar ruido) genera un eps dado."""
        test_labels = DBSCAN(eps=eps_val, min_samples=min_samples).fit_predict(X_weighted)
        return len(set(test_labels) - {-1})

    TARGET_MIN, TARGET_MAX = 3, 6
    n_clusters_codo = contar_clusters(eps_codo)

    if TARGET_MIN <= n_clusters_codo <= TARGET_MAX:
        return round(max(eps_codo, 0.1), 2)

    # --- Paso 3: búsqueda binaria sobre el rango de distancias reales ---
    eps_lo = float(distancias_ordenadas[0])             # más estricto
    eps_hi = float(distancias_ordenadas[-1])             # más permisivo
    eps_lo = max(eps_lo, 0.05)

    mejor_eps  = eps_codo
    mejor_diff = abs(n_clusters_codo - 4)

    for _ in range(25):  # máx 25 iteraciones de bisección
        eps_mid    = (eps_lo + eps_hi) / 2
        n_clusters = contar_clusters(eps_mid)

        diff = abs(n_clusters - 4)
        if diff < mejor_diff or (diff == mejor_diff and n_clusters >= TARGET_MIN):
            mejor_diff = diff
            mejor_eps  = eps_mid

        if TARGET_MIN <= n_clusters <= TARGET_MAX:
            return round(max(eps_mid, 0.1), 2)

        if n_clusters > TARGET_MAX:
            # demasiados clusters → eps muy chico → subir
            eps_lo = eps_mid
        elif n_clusters < TARGET_MIN:
            # muy pocos clusters → eps muy grande → bajar
            eps_hi = eps_mid
        else:
            break

        if (eps_hi - eps_lo) < 0.01:
            break

    return round(max(mejor_eps, 0.1), 2)



def guardar_archivos(modelo_obj: dict, metadatos: dict) -> None:
    """Persiste el modelo .pkl y los metadatos .json en el directorio de trabajo."""
    joblib.dump(modelo_obj, "modelo_arquetipos.pkl")
    with open("metadatos_modelo.json", "w", encoding="utf-8") as f:
        json.dump(metadatos, f, ensure_ascii=False, indent=2)



def interpretar_clusters(df_resultado: pd.DataFrame, labels) -> list[dict]:
    """
    Genera una lista de dicts con la interpretacion automatica de cada arquetipo.
    Cada dict tiene: arquetipo, n, pct, edad_prom, estres_prom,
                     patron_gasto, manejo_estres, color
    """
    unique_labels     = sorted(df_resultado["Cluster"].unique())
    arquetipos_reales = [l for l in unique_labels if l != -1]
    interpretaciones  = []

    for i, label in enumerate(arquetipos_reales):
        subset    = df_resultado[df_resultado["Cluster"] == label]
        n_arq     = len(subset)
        pct_arq   = round(n_arq / len(df_resultado) * 100, 1)
        edad_prom = round(subset["Edad"].mean(), 1)
        est_prom  = round(subset["P4_NivelEstres"].mean(), 1)
        p1_moda   = int(subset["P1_Destino_num"].mode()[0]) if n_arq > 0 else 0
        p2_moda   = int(subset["P2_Estres_num"].mode()[0])  if n_arq > 0 else 0
        color     = COLORES_PALETTE[i % len(COLORES_PALETTE)]

        interpretaciones.append({
            "arquetipo":     label,
            "n":             n_arq,
            "pct":           pct_arq,
            "edad_prom":     edad_prom,
            "estres_prom":   est_prom,
            "patron_gasto":  MAPA_P1.get(p1_moda, "Mixto"),
            "manejo_estres": MAPA_P2.get(p2_moda, "Mixto"),
            "color":         color,
        })

    return interpretaciones


def generar_reporte(
    nombre_alg: str,
    param_display: str,
    df_resultado: pd.DataFrame,
    interpretaciones: list[dict],
    varianza,
    filtro_genero: str,
    rango_edad: tuple,
) -> str:
    """Genera el texto del reporte listo para descargar como .txt / imprimir como PDF."""
    fecha_str = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    n_ruido   = int(list(df_resultado["Cluster"]).count(-1))

    lineas = ""
    for arq in interpretaciones:
        lineas += (
            f"\nArquetipo {arq['arquetipo']}  "
            f"({arq['n']} respondentes, {arq['pct']}%)\n"
            f"  - Edad promedio    : {arq['edad_prom']}\n"
            f"  - Estres promedio  : {arq['estres_prom']}/10\n"
            f"  - Patron de gasto  : {arq['patron_gasto']}\n"
            f"  - Manejo del estres: {arq['manejo_estres']}\n"
        )

    return f"""REPORTE DE RESULTADOS — ARQUETIPOS DE CONSUMO SOCIAL
============================================================
Fecha de generacion : {fecha_str}
Algoritmo utilizado : {nombre_alg}
Parametros          : {param_display}
Registros analizados: {len(df_resultado)}
Filtro de genero    : {filtro_genero}
Rango de edad       : {rango_edad[0]} - {rango_edad[1]} anos

------------------------------------------------------------
PONDERACION DE VARIABLES
------------------------------------------------------------
  P1 Destino del gasto       : 35%
  P2 Manejo del estres       : 25%
  P3 Frecuencia de salidas   : 20%
  P4 Nivel de estres (escala): 10%
  P5 Actividad fin de semana : 10%

Justificacion: la ponderacion diferencial se fundamenta en la
literatura de comportamiento del consumidor (Kotler, 2016) y
psicologia del estres (Lazarus & Folkman, 1984). Los pesos se
aplican post-estandarizacion para evitar sesgo por escala.

------------------------------------------------------------
ARQUETIPOS IDENTIFICADOS
------------------------------------------------------------
{lineas}
Puntos de ruido (atipicos): {n_ruido} ({round(n_ruido/len(df_resultado)*100, 1)}%)

------------------------------------------------------------
INTERPRETACION GENERAL
------------------------------------------------------------
El algoritmo {nombre_alg} opero sobre {len(df_resultado)} respuestas
ponderadas e identifico {len(interpretaciones)} arquetipo(s) de
consumo social estadisticamente cohesivos.

Los arquetipos representan agrupaciones naturales de perfiles
de comportamiento. El porcentaje de ruido valida la robustez
del modelo: respuestas inconsistentes son excluidas en lugar
de forzarse a un grupo.

Varianza explicada por PCA:
  Componente 1: {varianza[0]*100:.1f}%
  Componente 2: {varianza[1]*100:.1f}%
  Total        : {sum(varianza)*100:.1f}%

------------------------------------------------------------
Generado por: Aplicacion de Analisis No Supervisado (Streamlit)
============================================================
"""

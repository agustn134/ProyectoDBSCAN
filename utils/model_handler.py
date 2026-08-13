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

TARGET_MIN_CLUSTERS = 3
TARGET_MAX_CLUSTERS = 6


def min_samples_recomendado(n_registros: int, d_variables: int = 5) -> int:
    """
    Escala min_samples con el tamaño de muestra sin explotar en n pequeños.

    Regla: max(D+1, round(1% × n)), con techo en 50.

    Casos clave:
      n=73   → max(6, round(0.73)=1)  = 6   (regla clásica, funciona con datos reales)
      n=500  → max(6, round(5)=5)    = 6   (conservador para muestras medianas)
      n=5000 → max(6, round(50)=50)  = 50  (evita micro-clusters con datos sintéticos)

    La fórmula anterior (log(n)×D) daba ≈21 para n=73, lo que impide
    encontrar arquetipos válidos en muestras pequeñas.

    Parameters
    ----------
    n_registros : número de registros en el dataset
    d_variables : número de features usados (por defecto 5)

    Returns
    -------
    int con el min_samples recomendado (mínimo D+1, máximo 50)
    """
    base = d_variables + 1                    # piso clásico (6)
    proporcional = round(0.01 * n_registros)  # ~1% de la muestra
    valor = max(base, proporcional)
    return int(min(valor, 50))                # techo para n muy grande



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


def calcular_epsilon_sugerido(df: pd.DataFrame, min_samples: int | None = None) -> float:
    """
    Calcula el valor óptimo de epsilon mediante búsqueda en grilla log-espaciada.

    A diferencia de una bisección clásica, esta función NO asume que la curva
    #clusters vs eps sea monótona (en realidad tiene forma de campana: sube
    con eps pequeño y baja con eps grande). Una bisección puede quedar atrapada
    en el pico de micro-clusters; la grilla evalúa todo el rango de forma pareja.

    Cambios respecto a la versión anterior
    ---------------------------------------
    - min_samples se auto-calcula con min_samples_recomendado() si no se pasa.
      Esto evita que con 5000 registros min_samples=6 sea tan laxo que cualquier
      bolsita de puntos cuente como núcleo.
    - Se reemplaza la bisección (25 iteraciones) por una grilla de 40 candidatos
      en escala logarítmica sobre el rango real de distancias k-NN.

    Parameters
    ----------
    df          : DataFrame limpio con las columnas FEATURES
    min_samples : valor de min_samples a usar. Si es None se calcula
                  automáticamente con min_samples_recomendado().

    Returns
    -------
    float con el valor de epsilon sugerido, redondeado a 2 decimales
    """
    from sklearn.neighbors import NearestNeighbors

    n_registros = len(df)
    if min_samples is None:
        min_samples = min_samples_recomendado(n_registros)

    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_weighted = X_scaled * WEIGHTS

    k = min(max(min_samples - 1, 1), len(X_weighted) - 1)

    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X_weighted)
    distances, _ = neigh.kneighbors(X_weighted)
    d_ordenadas = np.sort(distances[:, k - 1])

    def contar_clusters(eps_val: float) -> int:
        """Retorna cuántos clusters (sin ruido) genera un eps dado."""
        test_labels = DBSCAN(eps=eps_val, min_samples=min_samples).fit_predict(X_weighted)
        return len(set(test_labels) - {-1})

    # --- Grilla log-espaciada: 40 candidatos sobre el rango real de distancias ---
    # geomspace garantiza cobertura uniforme en escala log, que es donde
    # la curva #clusters-vs-eps tiene más variación.
    d_min = max(float(d_ordenadas[0]), 0.02)
    d_max = float(d_ordenadas[-1])
    candidatos = np.unique(np.round(np.geomspace(d_min, d_max, 40), 3))

    resultados = [(float(eps), contar_clusters(eps)) for eps in candidatos]

    en_rango = [
        (e, c) for e, c in resultados
        if TARGET_MIN_CLUSTERS <= c <= TARGET_MAX_CLUSTERS
    ]

    if en_rango:
        # De los candidatos válidos, preferir el más cercano a 4 clusters
        mejor = min(en_rango, key=lambda ec: abs(ec[1] - 4))
    else:
        # Ningún candidato cayó en 3-6; devolver el menos malo
        mejor = min(resultados, key=lambda ec: abs(ec[1] - 4))

    return round(max(mejor[0], 0.05), 2)


def contar_clusters_para_eps(
    df: pd.DataFrame, eps: float, min_samples: int | None = None
) -> int:
    """
    Cuenta los clusters que produciría DBSCAN con los parámetros dados,
    sin guardar ni modificar nada. Útil para mostrar feedback en vivo en la UI.

    Parameters
    ----------
    df          : DataFrame limpio con las columnas FEATURES
    eps         : valor de epsilon a evaluar
    min_samples : si es None, se calcula con min_samples_recomendado()

    Returns
    -------
    int con el número de clusters encontrados (sin contar ruido)
    """
    if min_samples is None:
        min_samples = min_samples_recomendado(len(df))

    X = df[FEATURES].values
    X_weighted = StandardScaler().fit_transform(X) * WEIGHTS
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_weighted)
    return len(set(labels) - {-1})


def calcular_silueta(df: pd.DataFrame, labels) -> float | None:
    """
    Calcula el coeficiente de silueta promedio para los clusters encontrados,
    excluyendo los puntos de ruido (label == -1).

    El coeficiente de silueta mide qué tan bien separado está cada punto
    de los clusters vecinos respecto a su propio cluster. Escala [-1, 1]:
      > 0.5  → clusters bien definidos
      0.25-0.5 → estructura moderada
      < 0.25 → clusters solapados o estructura débil

    Parameters
    ----------
    df     : DataFrame filtrado (mismo que se usó para entrenar)
    labels : array de labels producido por DBSCAN / K-Means

    Returns
    -------
    float con el coeficiente, o None si no hay suficientes clusters.
    """
    from sklearn.metrics import silhouette_score

    labels_arr = np.array(labels)
    mask       = labels_arr != -1
    n_clusters = len(set(labels_arr[mask]))

    if n_clusters < 2 or mask.sum() < 4:
        return None

    X = df[FEATURES].values
    X_weighted = StandardScaler().fit_transform(X) * WEIGHTS

    try:
        return float(silhouette_score(X_weighted[mask], labels_arr[mask]))
    except Exception:
        return None


def calcular_kdistancia(
    df: pd.DataFrame, min_samples: int | None = None
) -> dict:
    """
    Calcula las k-distancias ordenadas para mostrar el gráfico del codo,
    que justifica visualmente el ε sugerido automáticamente.

    Parameters
    ----------
    df          : DataFrame limpio con las columnas FEATURES
    min_samples : si es None, se calcula con min_samples_recomendado()

    Returns
    -------
    dict con:
        distancias        : np.ndarray de k-distancias ordenadas (asc.)
        idx_codo          : int, índice del codo (máxima segunda derivada)
        eps_codo          : float, valor de eps en el codo
        min_samples_usado : int
    """
    from sklearn.neighbors import NearestNeighbors

    if min_samples is None:
        min_samples = min_samples_recomendado(len(df))

    X = df[FEATURES].values
    X_weighted = StandardScaler().fit_transform(X) * WEIGHTS

    k = min(max(min_samples - 1, 1), len(X_weighted) - 1)
    neigh = NearestNeighbors(n_neighbors=k).fit(X_weighted)
    distances, _ = neigh.kneighbors(X_weighted)
    d_sorted = np.sort(distances[:, k - 1])

    if len(d_sorted) > 2:
        segunda_deriv = np.diff(d_sorted, n=2)
        idx_codo = int(np.argmax(segunda_deriv)) + 2
    else:
        idx_codo = len(d_sorted) // 2

    return {
        "distancias":        d_sorted,
        "idx_codo":          idx_codo,
        "eps_codo":          float(d_sorted[idx_codo]),
        "min_samples_usado": min_samples,
    }



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

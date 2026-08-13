"""
app.py
------
Punto de entrada de la aplicacion Streamlit.
Este archivo SOLO orquesta la interfaz: no contiene logica de datos ni de ML.
  - Estilos   → utils/styles.py
  - Datos     → utils/data_handler.py
  - Modelo    → utils/model_handler.py
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go

from utils.styles       import inject_css
from utils.background   import inject_hero_geometric_background
from utils.data_handler import cargar_y_limpiar, aplicar_filtros, FEATURES
from utils.model_handler import (
    PESOS_INFO, COLORES_PALETTE,
    entrenar, guardar_archivos, interpretar_clusters, generar_reporte,
    calcular_epsilon_sugerido, contar_clusters_para_eps, min_samples_recomendado,
    calcular_silueta, calcular_kdistancia,
)

# ---------------------------------------------------------------------------
# Importaciones opcionales
# ---------------------------------------------------------------------------
try:
    import pygwalker as pyg
    PYGWALKER_OK = True
except ImportError:
    PYGWALKER_OK = False

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
    AGGRID_OK = True
except ImportError:
    AGGRID_OK = False

# ---------------------------------------------------------------------------
# CONFIGURACION DE PAGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Arquetipos de Consumo Social",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
inject_hero_geometric_background()

# ---------------------------------------------------------------------------
# HERO SECTION (Estilo Editorial Moderno)
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-section">
    <div class="pill-badge">
        <span>&#9673;</span> AI CLUSTERING &bull; ARQUETIPOS DE CONSUMO
    </div>
    <h1 class="hero-title">Patrones de Consumo</h1>
    <p class="hero-subtitle">Arquetipos de Personalidad en el Consumo Social.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Configuración")
uploaded_file = st.sidebar.file_uploader(
    "Cargar CSV (KoboToolbox)", type=["csv"], key="kobo_csv"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color:#64748b'>Separador: punto y coma ( ; )<br>"
    "Formato: 7 columnas.</small>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# SIN DATOS — pantalla de bienvenida
# ---------------------------------------------------------------------------
if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; background:#ffffff; border-radius:24px; border:1px solid #e5e7eb; margin:2rem auto; max-width:600px; box-shadow:0 4px 12px rgba(0,0,0,0.03);">
        <div style="font-size:3rem; margin-bottom:1rem; opacity:0.3;">&#9673;</div>
        <p style="font-size:1.3rem; font-weight:800; color:#0f0f11; margin-bottom:0.4rem;">Sin datos cargados</p>
        <p style="font-size:1.05rem; color:#64748b;">
            Carga tu archivo CSV desde el panel lateral para iniciar.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------------------
# CARGA Y PROCESAMIENTO
# ---------------------------------------------------------------------------
df = cargar_y_limpiar(uploaded_file)

n_total    = len(df)
n_generos  = df["Genero"].nunique()
edad_media = round(df["Edad"].mean(), 1)
estres_med = round(df["P4_NivelEstres"].mean(), 1)

# ---------------------------------------------------------------------------
# METRICAS
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-value">{n_total}</div>
        <div class="metric-label">Respuestas</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{n_generos}</div>
        <div class="metric-label">Géneros</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{edad_media}</div>
        <div class="metric-label">Edad promedio</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{estres_med}</div>
        <div class="metric-label">Estrés promedio (1-10)</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CONFIGURACION GLOBAL DE MATPLOTLIB (light theme editorial)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
    "axes.edgecolor":   "#e5e7eb", "axes.labelcolor": "#475569",
    "xtick.color":      "#64748b", "ytick.color":     "#64748b",
    "text.color":       "#0f0f11", "grid.color":      "#f1f5f9",
    "grid.linestyle":   "--",
})

# ===========================================================================
# SECCION 1 — DATOS Y FILTROS
# ===========================================================================
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

with st.expander("1.   Vista de datos y filtros", expanded=True):
    st.markdown("<p class='section-title'>Filtros de segmento</p>", unsafe_allow_html=True)
    col_f1, col_f2 = st.columns([1, 2])

    with col_f1:
        filtro_genero = st.selectbox(
            "Género",
            ["Todos"] + sorted(df["Genero"].dropna().unique().tolist()),
            key="fg",
        )
    with col_f2:
        edad_min = int(df["Edad"].min())
        edad_max = int(df["Edad"].max())
        filtro_edad = st.slider("Rango de edad", edad_min, edad_max,
                                (edad_min, edad_max), key="fe")

    df_filtrado = aplicar_filtros(df, filtro_genero, filtro_edad)

    st.markdown("<p class='section-title'>Respuestas filtradas</p>", unsafe_allow_html=True)
    cols_show = ["Edad", "Genero", "P1_Destino", "P2_Estres",
                 "P3_Frecuencia", "P4_NivelEstres", "P5_FinSemana"]
    df_vista = df_filtrado[cols_show].reset_index(drop=True)

    if AGGRID_OK:
        gb = GridOptionsBuilder.from_dataframe(df_vista)
        gb.configure_default_column(filter=True, sortable=True, resizable=True, editable=False)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
        gb.configure_grid_options(rowHeight=36, headerHeight=40)
        AgGrid(
            df_vista,
            gridOptions=gb.build(),
            update_mode=GridUpdateMode.NO_UPDATE,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            fit_columns_on_grid_load=True,
            theme="balham-dark",
            height=360,
        )
    else:
        st.dataframe(df_vista, use_container_width=True)

    st.caption(f"Mostrando {len(df_vista)} de {n_total} respuestas válidas.")

# ===========================================================================
# SECCION 2 — ESTADISTICA DESCRIPTIVA
# ===========================================================================
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

with st.expander("2.   Estadística descriptiva", expanded=False):
    st.markdown("<p class='section-title'>Distribución de variables</p>", unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**Distribución de edad**")
        fig1, ax1 = plt.subplots(figsize=(5, 3.2))
        sns.histplot(df_filtrado["Edad"], bins=10, kde=True, ax=ax1,
                     color="#0f0f11", edgecolor="#ffffff", alpha=0.85)
        ax1.set_xlabel("Edad"); ax1.set_ylabel("Frecuencia")
        ax1.yaxis.grid(True); ax1.set_axisbelow(True)
        fig1.tight_layout(); st.pyplot(fig1); plt.close(fig1)

    with col_s2:
        st.markdown("**Nivel de estrés percibido (1-10)**")
        fig2, ax2 = plt.subplots(figsize=(5, 3.2))
        sns.histplot(df_filtrado["P4_NivelEstres"], bins=10, kde=True, ax=ax2,
                     color="#27272a", edgecolor="#ffffff", alpha=0.85)
        ax2.set_xlabel("Nivel de estrés"); ax2.set_ylabel("Frecuencia")
        ax2.yaxis.grid(True); ax2.set_axisbelow(True)
        fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("**Composición por género**")
    col_s3, col_s4 = st.columns([1, 2])

    with col_s3:
        conteo_genero = df_filtrado["Genero"].value_counts().reset_index()
        conteo_genero.columns = ["Genero", "n"]
        st.dataframe(conteo_genero, use_container_width=True, hide_index=True)

    with col_s4:
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        color_map = {
            "Masculino": "#2563eb",
            "Femenino":  "#ec4899",
            "Otro":      "#10b981",
            "Non-binary": "#8b5cf6",
        }
        palette_fallback = ["#2563eb", "#ec4899", "#10b981", "#f59e0b", "#8b5cf6"]
        bar_colors = [
            color_map.get(str(g), palette_fallback[i % len(palette_fallback)])
            for i, g in enumerate(conteo_genero["Genero"])
        ]
        ax3.bar(conteo_genero["Genero"], conteo_genero["n"],
                color=bar_colors, edgecolor="#ffffff", linewidth=0.8, alpha=0.9)
        ax3.set_xlabel("Género"); ax3.set_ylabel("Respuestas")
        ax3.yaxis.grid(True); ax3.set_axisbelow(True)
        fig3.tight_layout(); st.pyplot(fig3); plt.close(fig3)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("**Resumen estadístico**")
    st.dataframe(df_filtrado[FEATURES].describe().round(2), use_container_width=True)

# ===========================================================================
# SECCION 3 — PONDERACION Y JUSTIFICACION TEORICA
# ===========================================================================
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

with st.expander("3.   Ponderación de variables — justificación teórica", expanded=True):
    st.markdown("<p class='section-title'>Esquema de pesos aplicado al modelo</p>",
                unsafe_allow_html=True)

    st.markdown("""<div class="justif-panel">
<p><strong>Criterio de Pesos:</strong> Las decisiones directas de gasto concentran el 60% del peso (P1 y P2). La frecuencia social y el nivel de estrés actúan como variables moduladoras (40%).</p>
<p>Se aplican <em>después</em> de la estandarización (Z-score). Suma total: 100%.</p>
</div>""", unsafe_allow_html=True)

    rows_html = ""
    for label, pct, justif in PESOS_INFO:
        rows_html += (
            f'<div class="weight-row">'
            f'<span class="weight-label">{label}</span>'
            f'<span class="weight-justif">{justif}</span>'
            f'<div class="weight-bar-wrap"><div class="weight-bar" style="width:{pct}%"></div></div>'
            f'<span class="weight-pct">{pct}%</span>'
            f'</div>'
        )
    st.markdown(f'<div class="weight-card">{rows_html}</div>', unsafe_allow_html=True)

    st.markdown(
        f"<small style='color:#64748b'>Muestra validada: {n_total} respuestas.</small>",
        unsafe_allow_html=True,
    )

# ===========================================================================
# SECCION 4 — ENTRENAMIENTO DEL MODELO
# ===========================================================================
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

with st.expander("4.   Entrenamiento del algoritmo de agrupamiento", expanded=True):
    st.markdown("<p class='section-title'>Selección de algoritmo y parámetros</p>",
                unsafe_allow_html=True)

    col_alg, col_e1, col_e2 = st.columns([1, 1, 1])

    with col_alg:
        algoritmo = st.selectbox(
            "Algoritmo de agrupamiento",
            ["DBSCAN (recomendado)", "K-Means"],
            index=0,
            key="alg",
            help="DBSCAN detecta clusters automáticamente y maneja ruido. "
                 "K-Means requiere definir el número de grupos manualmente.",
        )

    # Calcular min_samples y epsilon sugeridos con los datos filtrados
    MIN_SAMPLES_RECOMENDADO = min_samples_recomendado(len(df_filtrado))
    eps_sugerido = calcular_epsilon_sugerido(df_filtrado)  # usa min_samples auto

    if "DBSCAN" in algoritmo:
        with col_e1:
            eps = st.slider("Epsilon (ε) — radio de vecindad", 0.05, 10.0,
                            float(min(max(eps_sugerido, 0.05), 10.0)), 0.05, key="eps")

            # --- Preview en vivo: cuántos clusters produciría este eps ---
            n_preview = contar_clusters_para_eps(df_filtrado, eps, MIN_SAMPLES_RECOMENDADO)
            en_rango  = 3 <= n_preview <= 6
            badge_color  = "#16a34a" if en_rango else "#d97706"
            badge_bg     = "#f0fdf4" if en_rango else "#fffbeb"
            badge_border = "#bbf7d0" if en_rango else "#fde68a"
            badge_icon   = "✅" if en_rango else "⚠️"
            rango_txt    = "dentro del rango ideal (3-6)" if en_rango else "fuera del rango ideal (3-6)"
            st.markdown(
                f"""
                <div style='
                    display:inline-flex; align-items:center; gap:0.5rem;
                    background:{badge_bg}; border:1px solid {badge_border};
                    border-radius:8px; padding:0.4rem 0.85rem;
                    margin:0.4rem 0 0.2rem 0; font-size:0.88rem;
                '>
                    <span style='font-size:1.1rem;'>{badge_icon}</span>
                    <span>Con este ε, DBSCAN encontraría</span>
                    <strong style='color:{badge_color}; font-size:1.05rem;'>{n_preview}</strong>
                    <span>arquetipo{'s' if n_preview != 1 else ''} — <em>{rango_txt}</em></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style='font-size:0.82rem; color:#475569; margin-top:0.25rem; line-height:1.5;'>
                    <strong>¿Qué es?</strong> Define el radio de búsqueda alrededor de cada persona.
                    Dos personas con hábitos similares quedan dentro de este radio y se pueden agrupar.<br>
                    <strong>⚠️ Muy pequeño</strong> → muchos grupos diminutos o todo ruido.<br>
                    <strong>⚠️ Muy grande</strong> → todos terminan en un solo grupo.<br>
                    <span style='color:#3b82f6; font-weight:600;'>✦ Valor sugerido por tus datos: {eps_sugerido}</span>
                    <span style='color:#64748b;'> (calculado automáticamente con la técnica k-distancia)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_e2:
            min_samples = st.slider(
                "Mínimo de personas por grupo",
                2, max(30, MIN_SAMPLES_RECOMENDADO + 5),
                MIN_SAMPLES_RECOMENDADO,
                key="ms",
            )
            st.markdown(
                f"""
                <div style='font-size:0.82rem; color:#475569; margin-top:0.4rem; line-height:1.5;'>
                    <strong>¿Qué es?</strong> Cuántas personas como mínimo deben compartir hábitos
                    similares (estar dentro del radio ε) para que DBSCAN las considere un <em>arquetipo válido</em>.
                    Si hay menos, las marca como <strong>ruido</strong> (casos atípicos).<br>
                    <span style='color:#3b82f6; font-weight:600;'>✦ Valor sugerido: {MIN_SAMPLES_RECOMENDADO}</span>
                    <span style='color:#64748b;'> (escalado: max(6,&nbsp;1%&nbsp;×&nbsp;n={len(df_filtrado)}), techo 50)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        kw = dict(eps=eps, min_samples=min_samples)

        # --- Gráfico de k-distancia (justificación visual del ε sugerido) ---
        with st.expander("📐 Ver gráfico de k-distancia — cómo se calculó el ε sugerido", expanded=False):
            st.markdown(
                "<p style='color:#64748b; font-size:0.85rem; margin-bottom:0.6rem;'>"
                "Cada punto del eje Y es la distancia al <em>k</em>-ésimo vecino más cercano "
                "de una persona, ordenadas de menor a mayor. "
                "El <strong style='color:#ef4444;'>codo</strong> marca el cambio brusco entre "
                "'zona densa' y 'zona de ruido' — ahí está el ε óptimo."
                "</p>",
                unsafe_allow_html=True,
            )
            kd_data = calcular_kdistancia(df_filtrado, MIN_SAMPLES_RECOMENDADO)
            fig_kd, ax_kd = plt.subplots(figsize=(7, 3))
            ax_kd.plot(kd_data["distancias"], color="#3b82f6", linewidth=1.5, alpha=0.9,
                       label="k-distancia")
            ax_kd.axvline(kd_data["idx_codo"], color="#ef4444", linestyle="--",
                          linewidth=1.4, label=f"Codo → ε ≈ {kd_data['eps_codo']:.3f}")
            ax_kd.axhline(kd_data["eps_codo"], color="#ef4444", linestyle=":",
                          linewidth=1.0, alpha=0.45)
            ax_kd.scatter([kd_data["idx_codo"]], [kd_data["eps_codo"]],
                          color="#ef4444", s=55, zorder=5)
            ax_kd.set_xlabel("Puntos ordenados", fontsize=9)
            ax_kd.set_ylabel(f"{kd_data['min_samples_usado']}-distancia (ε)", fontsize=9)
            ax_kd.set_title("Gráfico de k-distancia para selección de ε", fontsize=10)
            ax_kd.legend(fontsize=8); ax_kd.yaxis.grid(True); ax_kd.set_axisbelow(True)
            fig_kd.tight_layout()
            st.pyplot(fig_kd)
            plt.close(fig_kd)
            st.caption(
                f"ε sugerido: {eps_sugerido}  ·  min_samples usado: {kd_data['min_samples_usado']}  ·  "
                f"n = {len(df_filtrado)} registros"
            )
    else:
        with col_e1:
            k_clusters = st.slider("Número de clusters (k)", 2, 8, 3, key="kc")
            st.markdown(
                "<div style='font-size:0.82rem; color:#475569; margin-top:0.4rem;'>"
                "Cantidad de arquetipos que quieres que el algoritmo encuentre. "
                "A diferencia de DBSCAN, K-Means siempre crea exactamente este número de grupos."
                "</div>",
                unsafe_allow_html=True,
            )
        with col_e2:
            st.markdown(
                "<div style='padding-top:1.8rem; color:#64748b; font-size:0.82rem;'>"
                "K-Means asigna <strong>todos</strong> los registros a algún grupo. "
                "No genera puntos de ruido, por lo que personas muy atípicas igual "
                "quedarán forzadas dentro de un arquetipo.</div>",
                unsafe_allow_html=True,
            )
        kw = dict(k_clusters=k_clusters)

    st.markdown("<br>", unsafe_allow_html=True)
    run_model = st.button("Entrenar modelo y generar resultados")

    if run_model:
        if len(df_filtrado) < 4:
            st.error("No hay suficientes registros con los filtros actuales.")
            st.stop()

        with st.spinner("Estandarizando, ponderando y ejecutando algoritmo..."):
            res = entrenar(df_filtrado, algoritmo, **kw)
            guardar_archivos(res["modelo_obj"], res["metadatos"])
            st.session_state["modelo_res"] = res

    if "modelo_res" in st.session_state:
        res = st.session_state["modelo_res"]
        st.success(
            f"Modelo entrenado ({res['nombre_alg']}) y guardado automáticamente en el proyecto para el entrenamiento de datos."
        )

        # -----------------------------------------------------------------------
        # Tabla de resultados
        # -----------------------------------------------------------------------
        st.markdown("<p class='section-title'>Resultados del agrupamiento</p>",
                    unsafe_allow_html=True)

        conteo = res["df_resultado"]["Cluster"].value_counts().reset_index()
        conteo.columns = ["Cluster_ID", "Cantidad"]
        conteo["Arquetipo"] = conteo["Cluster_ID"].apply(
            lambda x: "Ruido (atípicos)" if x == -1 else f"Arquetipo {x}"
        )
        conteo["Porcentaje"] = (
            conteo["Cantidad"] / len(res["df_resultado"]) * 100
        ).round(1).astype(str) + "%"
        st.dataframe(
            conteo[["Arquetipo", "Cantidad", "Porcentaje"]],
            use_container_width=True, hide_index=True,
        )

        # -----------------------------------------------------------------------
        # Interpretaciones + Nombres teóricos (calculadas ANTES del plot)
        # -----------------------------------------------------------------------
        # --- Coeficiente de silueta ---
        sil = calcular_silueta(df_filtrado, res["labels"])
        if sil is not None:
            sil_color  = "#16a34a" if sil >= 0.5 else ("#d97706" if sil >= 0.25 else "#dc2626")
            sil_bg     = "#f0fdf4" if sil >= 0.5 else ("#fffbeb" if sil >= 0.25 else "#fef2f2")
            sil_border = "#bbf7d0" if sil >= 0.5 else ("#fde68a" if sil >= 0.25 else "#fecaca")
            sil_label  = "Bueno ✅" if sil >= 0.5 else ("Moderado ⚠️" if sil >= 0.25 else "Débil ❌")
            sil_interp = (
                "Los arquetipos están bien definidos y claramente separados entre sí."
                if sil >= 0.5 else
                "Los arquetipos tienen estructura razonable pero con cierto solapamiento entre perfiles."
                if sil >= 0.25 else
                "Los arquetipos se solapan considerablemente. Considera ajustar ε o min_samples."
            )
            st.markdown(
                f"""
                <div style='background:{sil_bg}; border:1px solid {sil_border};
                            border-left:4px solid {sil_color}; border-radius:10px;
                            padding:0.75rem 1.15rem; margin:0.5rem 0 0.8rem 0;'>
                    <span style='font-size:0.88rem; color:#374151;'>
                        <strong>Coeficiente de Silueta</strong>
                        &nbsp;&bull;&nbsp;
                        <span style='font-size:1.35rem; font-weight:800;
                                     color:{sil_color};'>{sil:.3f}</span>
                        &nbsp;—&nbsp;
                        <strong style='color:{sil_color};'>{sil_label}</strong>
                    </span><br>
                    <span style='font-size:0.82rem; color:#475569;'>
                        {sil_interp}
                        &nbsp;<em>(Escala: −1 pésimo &bull; 0 solapado &bull; +1 perfecto)</em>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        from scipy.spatial import ConvexHull

        df_resultado = res["df_resultado"]
        labels_arr   = res["labels"]

        def determinar_nombre_arquetipo(subset):
            """
            Asigna nombre teórico usando P1 (gasto), P2 (manejo estrés),
            P3 (frecuencia) y P4 (nivel estrés) para mayor especificidad
            y menor riesgo de colisión entre clusters distintos.
            """
            if len(subset) == 0:
                return "Perfil Mixto", "#6366f1", "Sin datos suficientes."
            p1_moda    = int(subset["P1_Destino_num"].mode()[0])
            p2_moda    = int(subset["P2_Estres_num"].mode()[0])
            p3_mediana = subset["P3_Frecuencia_num"].median()
            p4_media   = subset["P4_NivelEstres"].mean()

            if p1_moda == 1:
                # Gasta en entretenimiento/salidas
                if p4_media >= 6.5:
                    return (
                        "Hedonista Social — Alto Estrés", "#3b82f6",
                        "Sale frecuentemente a desconectarse con alcohol y fiestas. "
                        "El consumo social es su válvula de escape frente a un estrés elevado.",
                    )
                else:
                    return (
                        "Hedonista Social — Bajo Estrés", "#60a5fa",
                        "Gasta en salidas y ocio desde un estado de bienestar. "
                        "El consumo social es parte de su estilo de vida, no una compensación.",
                    )
            elif p1_moda == 2:
                # Gasta en bienestar / salud
                if p2_moda == 1:   # manejo activo (ejercicio)
                    return (
                        "Bienestar Consciente — Activo", "#10b981",
                        "Invierte en gym, comida sana o naturaleza y maneja el estrés "
                        "de forma proactiva con actividad física.",
                    )
                else:
                    return (
                        "Bienestar Consciente — Reflexivo", "#34d399",
                        "Orienta su gasto hacia el equilibrio personal. "
                        "Prefiere descansar o planificar antes de salir a consumir.",
                    )
            elif p1_moda == 3 and p3_mediana <= 2:
                return (
                    "Equilibrado Práctico", "#f59e0b",
                    "Gasta de forma moderada y prioriza el ahorro o metas a futuro. "
                    "Sale poco y cuando lo hace elige actividades de bajo costo.",
                )
            elif p1_moda == 4 or (p1_moda == 3 and p3_mediana > 2):
                if p3_mediana >= 3:
                    return (
                        "Explorador de Experiencias — Frecuente", "#8b5cf6",
                        "Sale seguido buscando novedad: viajes, cursos o aventuras. "
                        "Gasta en crecimiento y descubrimiento, no en rutina.",
                    )
                else:
                    return (
                        "Explorador de Experiencias — Ocasional", "#a78bfa",
                        "Busca experiencias de forma selectiva. "
                        "Calidad sobre cantidad: pocas salidas pero significativas.",
                    )
            else:
                return (
                    "Perfil Mixto", "#64748b",
                    "Combinación de patrones sin predominancia clara en ninguna dimensión.",
                )

        unique_labels     = sorted(df_resultado["Cluster"].unique())
        arquetipos_reales = [l for l in unique_labels if l != -1]
        v = res["varianza"]

        # Construir mapa label -> metadatos del arquetipo
        info_arq = {}
        interpretaciones = []
        for label in arquetipos_reales:
            subset = df_resultado[df_resultado["Cluster"] == label]
            nombre, color, desc = determinar_nombre_arquetipo(subset)
            n_arq       = len(subset)
            pct_arq     = round(n_arq / len(df_resultado) * 100, 1) if len(df_resultado) > 0 else 0
            edad_prom   = round(subset["Edad"].mean(), 1) if n_arq > 0 else 0
            estres_prom = round(subset["P4_NivelEstres"].mean(), 1) if n_arq > 0 else 0
            p1_str      = subset["P1_Destino"].mode()[0] if n_arq > 0 else "N/A"
            p2_str      = subset["P2_Estres"].mode()[0]  if n_arq > 0 else "N/A"
            info_arq[label] = {
                "nombre": nombre, "color": color, "desc": desc,
                "n": n_arq, "pct": pct_arq,
                "edad_prom": edad_prom, "estres_prom": estres_prom,
            }
            interpretaciones.append({
                "arquetipo": f"{label} - {nombre}",
                "n": n_arq, "pct": pct_arq,
                "edad_prom": edad_prom, "estres_prom": estres_prom,
                "patron_gasto": p1_str, "manejo_estres": p2_str,
            })

        # --- Garantizar nombres únicos por cluster ---
        # Si dos clusters comparten nombre (misma regla P1), se distinguen
        # por nivel de estrés promedio + ID para evitar confusión en el reporte.
        nombre_a_labels: dict = {}
        for lbl in arquetipos_reales:
            nm = info_arq[lbl]["nombre"]
            nombre_a_labels.setdefault(nm, []).append(lbl)
        for nm, lbls in nombre_a_labels.items():
            if len(lbls) > 1:
                for lbl in lbls:
                    est = info_arq[lbl]["estres_prom"]
                    sufijo = "Alto Estrés" if est >= 5.5 else "Bajo Estrés"
                    info_arq[lbl]["nombre"] = f"{nm} · {sufijo} (A{lbl})"

        # -----------------------------------------------------------------------
        # Grafica PCA interactiva con Convex Hull
        # -----------------------------------------------------------------------
        st.markdown("**🗺️ Mapa interactivo de arquetipos de consumo social**")
        st.caption(
            "Pasa el cursor sobre cada punto para ver los detalles · "
            "Haz zoom arrastrando · Haz clic en la leyenda para ocultar/mostrar grupos"
        )

        fig_plotly = go.Figure()

        def hex_to_rgba(hex_color: str, alpha: float = 0.10) -> str:
            """Convierte '#3b82f6' a 'rgba(59,130,246,0.10)'."""
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        # --- Contornos Convex Hull (se agregan primero para quedar detrás) ---
        for label in arquetipos_reales:
            subset = df_resultado[df_resultado["Cluster"] == label]
            pts = subset[["PCA_1", "PCA_2"]].values
            color = info_arq[label]["color"]
            nombre = info_arq[label]["nombre"]

            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    hull_pts = pts[hull.vertices]
                    hull_pts = list(hull_pts) + [hull_pts[0]]   # cerrar polígono
                    hx = [p[0] for p in hull_pts]
                    hy = [p[1] for p in hull_pts]

                    fig_plotly.add_trace(go.Scatter(
                        x=hx, y=hy,
                        fill="toself",
                        fillcolor=hex_to_rgba(color, 0.10),
                        line=dict(color=color, width=1.8, dash="dot"),
                        mode="lines",
                        name=nombre,
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                except Exception:
                    pass  # si hay colinealidad, se salta el hull

        # --- Puntos de cada cluster (Scattergl para rendimiento + jitter visual) ---
        # El jitter (ruido aleatorio pequeño) revela puntos superpuestos en datasets
        # con variables categóricas, donde muchas filas comparten coordenadas PCA exactas.
        # Es puramente visual: no altera el clustering ni los datos originales.
        pca1_rng = df_resultado["PCA_1"].max() - df_resultado["PCA_1"].min() + 1e-6
        pca2_rng = df_resultado["PCA_2"].max() - df_resultado["PCA_2"].min() + 1e-6
        jitter_pct = 0.008  # 0.8% del rango total

        for i, label in enumerate(unique_labels):
            subset = df_resultado[df_resultado["Cluster"] == label].copy()

            if label == -1:
                color        = "#a1a1aa"
                nombre_grupo = "⬡ Casos atípicos / Ruido"
            else:
                color        = info_arq[label]["color"]
                nombre_grupo = f"{info_arq[label]['nombre']} (A{label})"

            # Jitter reproducible por subset (misma seed → mismo dibujo en cada recarga)
            rng_j  = np.random.default_rng(42 + (label if label != -1 else 999))
            x_plot = subset["PCA_1"].values + rng_j.uniform(
                -jitter_pct * pca1_rng, jitter_pct * pca1_rng, len(subset)
            )
            y_plot = subset["PCA_2"].values + rng_j.uniform(
                -jitter_pct * pca2_rng, jitter_pct * pca2_rng, len(subset)
            )

            hover_texts = []
            for _, row in subset.iterrows():
                nombre_h = info_arq[label]["nombre"] if label != -1 else "Caso atípico"
                hover_texts.append(
                    f"<b>{nombre_h}</b><br>"
                    f"Edad: {int(row['Edad'])} años<br>"
                    f"Estrés: {row['P4_NivelEstres']}/10<br>"
                    f"Género: {row['Genero']}<br>"
                    f"Gasto: {row.get('P1_Destino', '—')}<br>"
                    f"Manejo estrés: {row.get('P2_Estres', '—')}"
                )

            fig_plotly.add_trace(go.Scattergl(
                x=x_plot, y=y_plot,
                mode="markers",
                name=nombre_grupo,
                marker=dict(color=color, size=7, opacity=0.72,
                            line=dict(color="white", width=0.6)),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_texts,
            ))

            # Etiqueta de centroide
            if label != -1:
                cx, cy = subset["PCA_1"].mean(), subset["PCA_2"].mean()
                fig_plotly.add_annotation(
                    x=cx, y=cy,
                    text=f"<b>{info_arq[label]['nombre']}</b>",
                    showarrow=False,
                    font=dict(size=10, color=color),
                    bgcolor="white", bordercolor=color,
                    borderwidth=1.5, borderpad=4, opacity=0.92,
                )

        fig_plotly.update_layout(
            title=dict(
                text=f"Mapa de arquetipos de consumo social — {res['nombre_alg']}",
                font=dict(size=14, color="#0f0f11"), x=0,
            ),
            xaxis=dict(
                title=dict(
                    text=f"Eje de Gasto y Frecuencia social ({v[0]*100:.1f}% de variación)",
                    font=dict(size=11, color="#475569"),
                ),
                tickfont=dict(size=9, color="#94a3b8"),
                gridcolor="#f1f5f9", showgrid=True,
                zeroline=True, zerolinecolor="#e2e8f0", zerolinewidth=1.5,
            ),
            yaxis=dict(
                title=dict(
                    text=f"Eje de Estrés y Actividad ({v[1]*100:.1f}% de variación)",
                    font=dict(size=11, color="#475569"),
                ),
                tickfont=dict(size=9, color="#94a3b8"),
                gridcolor="#f1f5f9", showgrid=True,
                zeroline=True, zerolinecolor="#e2e8f0", zerolinewidth=1.5,
            ),
            plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
            legend=dict(
                title=dict(text="Arquetipos", font=dict(size=11)),
                font=dict(size=10, color="#374151"),
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#e2e8f0", borderwidth=1,
                orientation="v", x=1.01, xanchor="left", y=1, yanchor="top",
            ),
            hoverlabel=dict(bgcolor="white", bordercolor="#e2e8f0",
                            font=dict(size=12, color="#1e293b")),
            margin=dict(l=60, r=240, t=60, b=60),
            height=540,
        )

        st.plotly_chart(fig_plotly, use_container_width=True)

        st.markdown(
            """
            <div style='font-size:0.82rem; color:#64748b; background:#f8fafc;
                        border-left:3px solid #cbd5e1; padding:0.65rem 1rem;
                        border-radius:0 8px 8px 0; margin-top:0.2rem;'>
            <strong>📌 ¿Por qué los ejes tienen números como −0.4 o 0.6?</strong><br>
            Este mapa no muestra edades ni respuestas directas. El algoritmo combinó tus 5 variables
            en 2 "resúmenes". Los números son <strong>distancias relativas</strong>: lo importante
            no es el número, sino <em>qué tan cerca o lejos</em> están los puntos entre sí.
            El <strong>contorno punteado</strong> de cada color muestra la "frontera" del arquetipo.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -----------------------------------------------------------------------
        # Interpretación por arquetipo — con nombres teóricos
        # -----------------------------------------------------------------------
        st.markdown("<p class='section-title'>Interpretación por arquetipo</p>",
                    unsafe_allow_html=True)

        for label in arquetipos_reales:
            a = info_arq[label]
            st.markdown(
                f"""<div class="justif-panel" style="border-left-color:{a['color']};">
<p><strong>{a['nombre']} &nbsp;(Arquetipo {label})</strong>
&nbsp;|&nbsp; {a['n']} respondentes ({a['pct']}%)
&nbsp;|&nbsp; Edad promedio: {a['edad_prom']}
&nbsp;|&nbsp; Estrés promedio: {a['estres_prom']}/10</p>
<p><em>{a['desc']}</em></p>
</div>""",
                unsafe_allow_html=True,
            )

        n_ruido = int(list(labels_arr).count(-1))
        if n_ruido > 0:
            st.markdown(
                f"""<div class="justif-panel" style="border-left-color:#71717a;">
<p><strong>⬡ Casos atípicos (Ruido)</strong>
&nbsp;|&nbsp; {n_ruido} respondentes ({round(n_ruido/len(df_resultado)*100, 1)}%)</p>
<p>Perfiles con respuestas inconsistentes que no forman densidad suficiente para pertenecer
a ningún arquetipo. Su presencia <strong>valida la robustez del modelo</strong>.</p>
</div>""",
                unsafe_allow_html=True,
            )

        st.info(
            f"El algoritmo {res['nombre_alg']} identificó {len(arquetipos_reales)} arquetipo(s) "
            f"con parámetros: {res['param_display']}."
        )

        # -----------------------------------------------------------------------
        # Exportar — Bloque Banner Oscuro estilo CTA
        # -----------------------------------------------------------------------
        st.markdown("""<div class="dark-cta-banner">
<h2>Exportar Resultados &amp; Modelo</h2>
<p>Descarga los archivos generados por el algoritmo para su integración en producción o análisis posterior.</p>
</div>""", unsafe_allow_html=True)

        col_d1, col_d2, col_d3, col_d4 = st.columns(4)

        with col_d1:
            st.download_button(
                label="Datos (CSV)",
                data=res["df_resultado"].to_csv(index=False).encode("utf-8"),
                file_name="resultados_arquetipos.csv",
                mime="text/csv",
            )
        with col_d2:
            with open("modelo_arquetipos.pkl", "rb") as f_pkl:
                st.download_button(
                    label="Modelo (.pkl)",
                    data=f_pkl,
                    file_name="modelo_arquetipos.pkl",
                    mime="application/octet-stream",
                )
        with col_d3:
            with open("metadatos_modelo.json", "rb") as f_json:
                st.download_button(
                    label="Metadatos (.json)",
                    data=f_json,
                    file_name="metadatos_modelo.json",
                    mime="application/json",
                )
        with col_d4:
            reporte = generar_reporte(
                res["nombre_alg"], res["param_display"],
                res["df_resultado"], interpretaciones,
                res["varianza"], filtro_genero, filtro_edad,
            )
            st.download_button(
                label="Reporte (.txt)",
                data=reporte.encode("utf-8"),
                file_name="reporte_arquetipos.txt",
                mime="text/plain",
            )

# ===========================================================================
# SECCION 5 — EXPLORADOR VISUAL INTERACTIVO (PYGWALKER)
# ===========================================================================
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

with st.expander("5.   Explorador visual interactivo", expanded=False):
    if not PYGWALKER_OK:
        st.warning("Pygwalker no está instalado. Ejecuta: pip install pygwalker")
    else:
        st.markdown(
            "<p style='color:#64748b; font-size:0.88rem; margin-bottom:1rem;'>"
            "Arrastra variables hacia los ejes X / Y para construir visualizaciones personalizadas. "
            "Soporta gráficas de dispersión, barras, histogramas, mapas de calor y más."
            "</p>",
            unsafe_allow_html=True,
        )

        cols_pyg = ["Edad", "Genero", "P1_Destino_num", "P2_Estres_num",
                    "P3_Frecuencia_num", "P4_NivelEstres", "P5_FinSemana_num"]
        df_pyg = df_filtrado[cols_pyg].copy()
        df_pyg.columns = ["Edad", "Genero", "P1_Destino", "P2_Estres",
                          "P3_Frecuencia", "P4_NivelEstres", "P5_FinSemana"]

        try:
            html_content = pyg.to_html(df_pyg, appearance="light", theme_key="streamlit", return_html=True)
            components.html(html_content, height=700, scrolling=True)
        except Exception as e:
            try:
                pyg.walk(df_pyg, env="Streamlit", appearance="light")
            except Exception:
                st.error(
                    f"No se pudo inicializar el explorador visual. "
                    f"Detalle del error: {e}"
                )

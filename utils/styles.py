"""
utils/styles.py
---------------
Centraliza todos los estilos CSS de la aplicacion.
Tipografia optimizada de alta legibilidad ('Plus Jakarta Sans'),
tamaños de fuente ampliados para proyeccion y lectura clara.
"""

import streamlit as st

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { 
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #f4f4f6 !important; 
    color: #0f172a;
    font-size: 17px;
}

/* El fondo principal de Streamlit debe ser transparente para ver el WebGL/fondo debajo */
.stApp { 
    background-color: transparent !important; 
    color: #0f172a; 
}
.stAppHeader { 
    background-color: transparent !important; 
}

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
    box-shadow: 2px 0 10px rgba(0,0,0,0.02);
}
section[data-testid="stSidebar"] * { 
    color: #1e293b !important; 
    font-size: 1rem !important;
}

/* ---- Pill Badges ---- */
.pill-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    background: #e2e8f0;
    color: #0f172a;
    font-size: 0.88rem;
    font-weight: 700;
    padding: 0.45rem 1.1rem;
    border-radius: 9999px;
    margin-bottom: 1.2rem;
    letter-spacing: -0.01em;
}

/* ---- Hero Section (Estilo Editorial) ---- */
.hero-section {
    position: relative;
    padding: 3.8rem 1.5rem 2.8rem 1.5rem;
    text-align: center;
    max-width: 980px;
    margin: 0 auto 1.5rem auto;
    animation: fadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}
.hero-title {
    font-size: 3.8rem;
    font-weight: 800;
    color: #0f0f11;
    line-height: 1.1;
    letter-spacing: -0.04em;
    margin-bottom: 1.2rem;
}
.hero-subtitle {
    font-size: 1.25rem;
    color: #475569;
    font-weight: 500;
    max-width: 680px;
    width: 100%;
    margin: 0 auto 2rem auto;
    line-height: 1.5;
    text-align: center !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ---- Metricas en Tarjetas Redondeadas ---- */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.4rem;
    margin: 2rem 0;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    transition: transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}
.metric-card:hover { 
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(0,0,0,0.06); 
    border-color: #cbd5e1;
}
.metric-value {
    font-size: 3rem;
    font-weight: 800;
    color: #0f0f11;
    line-height: 1;
    letter-spacing: -0.03em;
}
.metric-label {
    font-size: 1rem;
    color: #475569;
    margin-top: 0.6rem;
    font-weight: 600;
}

/* ---- Titulos de seccion ---- */
.section-title {
    font-size: 1.2rem;
    font-weight: 800;
    color: #0f0f11;
    margin: 1.6rem 0 1rem 0;
    letter-spacing: -0.02em;
}

/* ---- Tarjeta de Ponderacion ---- */
.weight-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 1.6rem 2rem;
    margin: 1.2rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}
.weight-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 1rem;
}
.weight-row:last-child { border-bottom: none; }
.weight-label { color: #0f0f11; flex: 0 0 240px; font-weight: 700; font-size: 1.05rem; }
.weight-justif {
    color: #475569;
    font-size: 0.95rem;
    flex: 1;
    padding: 0 1.4rem;
}
.weight-bar-wrap {
    flex: 0 0 140px;
    background: #e2e8f0;
    border-radius: 9999px;
    height: 10px;
    margin-right: 1rem;
    overflow: hidden;
}
.weight-bar {
    background: #0f0f11;
    height: 100%;
    border-radius: 9999px;
}
.weight-pct {
    font-weight: 800;
    color: #0f0f11;
    font-size: 1.05rem;
    min-width: 45px;
    text-align: right;
    flex: 0 0 45px;
}

/* ---- Panel de Justificacion Teórica ---- */
.justif-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 5px solid #0f0f11;
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    margin: 1.2rem 0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02);
}
.justif-panel p {
    font-size: 1rem;
    color: #334155;
    margin: 0.5rem 0;
    line-height: 1.6;
}

/* ---- Botones Estilo Píldora (Negro Sólido) ---- */
.stButton > button {
    background: #0f0f11 !important;
    color: #ffffff !important;
    border: 1px solid #0f0f11 !important;
    border-radius: 9999px !important;
    padding: 0.75rem 2.2rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1) !important;
}
.stButton > button:hover {
    background: #27272a !important;
    border-color: #27272a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 14px rgba(0,0,0,0.18) !important;
}

/* ---- Expanders (Tarjetas Blancas Redondeadas) ---- */
.streamlit-expanderHeader {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    color: #0f0f11 !important;
    font-weight: 700 !important;
    font-size: 1.12rem !important;
    padding: 1.1rem 1.6rem !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02) !important;
}
.streamlit-expanderContent {
    border: 1px solid #e2e8f0 !important;
    border-top: none !important;
    border-radius: 0 0 16px 16px !important;
    background: #ffffff !important;
    padding: 1.6rem !important;
}

/* ---- Banner Oscuro CTA / Cierre ---- */
.dark-cta-banner {
    background: #0f0f11;
    color: #ffffff;
    border-radius: 24px;
    padding: 2.8rem 2.2rem;
    margin: 3rem 0 1.5rem 0;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
}
.dark-cta-banner h2 {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.8rem;
    letter-spacing: -0.03em;
}
.dark-cta-banner p {
    color: #a1a1aa;
    font-size: 1.1rem;
    max-width: 620px;
    margin: 0 auto 1.8rem auto;
}

/* ---- Separador Editorial ---- */
.section-divider {
    height: 1px;
    background: #e2e8f0;
    margin: 2.5rem 0;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f4f4f6; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 9999px; }
"""


def inject_css() -> None:
    """Inyecta los estilos CSS globales en la pagina de Streamlit."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

"""
generar_datos_realista.py
--------------------------
Genera un CSV sintetico de respuestas para el proyecto "Arquetipos de
Consumo Social", con variabilidad realista en lugar de respuestas
uniformemente aleatorias.

Uso:
    python generar_datos_realista.py --n 5000 --salida datos_5000_v2.csv
"""

import argparse
import numpy as np
import pandas as pd

ARQUETIPOS = {
    "Hedonista Social": dict(
        probs_p1=[0.55, 0.20, 0.15, 0.10],
        probs_p2=[0.10, 0.55, 0.20, 0.15],
        probs_p3=[0.50, 0.30, 0.15, 0.05],
        probs_p5=[0.55, 0.25, 0.10, 0.10],
        estres_media=6.8, estres_std=1.4,
        edad_media=22, edad_std=3,
        peso=0.28,
    ),
    "Bienestar Consciente": dict(
        probs_p1=[0.15, 0.10, 0.55, 0.20],
        probs_p2=[0.60, 0.10, 0.15, 0.15],
        probs_p3=[0.10, 0.20, 0.40, 0.30],
        probs_p5=[0.10, 0.15, 0.55, 0.20],
        estres_media=4.0, estres_std=1.3,
        edad_media=27, edad_std=5,
        peso=0.24,
    ),
    "Explorador de Experiencias": dict(
        probs_p1=[0.15, 0.15, 0.15, 0.55],
        probs_p2=[0.15, 0.20, 0.15, 0.50],
        probs_p3=[0.30, 0.35, 0.25, 0.10],
        probs_p5=[0.20, 0.50, 0.15, 0.15],
        estres_media=5.2, estres_std=1.6,
        edad_media=24, edad_std=4,
        peso=0.22,
    ),
    "Equilibrado Practico": dict(
        probs_p1=[0.20, 0.15, 0.50, 0.15],
        probs_p2=[0.20, 0.15, 0.50, 0.15],
        probs_p3=[0.10, 0.15, 0.55, 0.20],
        probs_p5=[0.15, 0.15, 0.55, 0.15],
        estres_media=4.6, estres_std=1.2,
        edad_media=30, edad_std=6,
        peso=0.18,
    ),
    "Ansioso Compensador": dict(
        probs_p1=[0.30, 0.10, 0.20, 0.40],
        probs_p2=[0.05, 0.65, 0.10, 0.20],
        probs_p3=[0.40, 0.35, 0.15, 0.10],
        probs_p5=[0.45, 0.20, 0.10, 0.25],
        estres_media=8.1, estres_std=1.2,
        edad_media=23, edad_std=3,
        peso=0.08,
    ),
}

RUIDO_INTRA   = 0.18
PROP_OUTLIERS = 0.04

OPCIONES_LETRA = ["A", "B", "C", "D"]
TEXTO_P1 = {
    "A": "A) Salir a beber con amigos y fiestas.",
    "B": "B) Entretenimiento (streaming, videojuegos, salidas).",
    "C": "C) Necesidades basicas y ahorro.",
    "D": "D) Experiencias, viajes o aprendizaje.",
}
TEXTO_P2 = {
    "A": "A) Hacer ejercicio y meditar.",
    "B": "B) Salir a desconectarme, beber y socializar.",
    "C": "C) Descansar en casa, planeando mi tiempo y gastos.",
    "D": "D) Consumo digital (redes, series, compras online).",
}
TEXTO_P3 = {
    "A": "A) 4 o mas veces.",
    "B": "B) 2 a 3 veces.",
    "C": "C) 1 vez.",
    "D": "D) Solo cuando hay un evento importante.",
}
TEXTO_P5 = {
    "A": "A) Improvisado, con amigos y musica.",
    "B": "B) Explorando algo nuevo (lugar, actividad, curso).",
    "C": "C) Tranquilo, en casa y preparando la semana.",
    "D": "D) Depende del dinero disponible esa semana.",
}
GENEROS = ["Hombre", "Mujer"]


def _muestrear_categoria(probs, ruido, rng):
    if rng.random() < ruido:
        idx = rng.integers(0, 4)
    else:
        idx = rng.choice(4, p=probs)
    return OPCIONES_LETRA[int(idx)]


def _normal_truncada(media, std, lo, hi, rng):
    return float(np.clip(rng.normal(media, std), lo, hi))


def generar_dataset(n, seed=42):
    rng = np.random.default_rng(seed)
    nombres = list(ARQUETIPOS.keys())
    pesos   = np.array([ARQUETIPOS[nm]["peso"] for nm in nombres])
    pesos   = pesos / pesos.sum()
    n_outliers  = int(n * PROP_OUTLIERS)
    n_arquetipo = n - n_outliers
    asignaciones = rng.choice(nombres, size=n_arquetipo, p=pesos)
    filas = []

    for arq_nombre in asignaciones:
        arq = ARQUETIPOS[arq_nombre]
        p1 = _muestrear_categoria(arq["probs_p1"], RUIDO_INTRA, rng)
        p2 = _muestrear_categoria(arq["probs_p2"], RUIDO_INTRA, rng)
        p3 = _muestrear_categoria(arq["probs_p3"], RUIDO_INTRA, rng)
        p5 = _muestrear_categoria(arq["probs_p5"], RUIDO_INTRA, rng)
        estres = int(round(_normal_truncada(arq["estres_media"], arq["estres_std"], 1, 10, rng)))
        edad   = int(round(_normal_truncada(arq["edad_media"],   arq["edad_std"],   18, 45, rng)))
        genero = str(rng.choice(GENEROS, p=[0.48, 0.52]))
        filas.append(dict(
            Edad=edad, Genero=genero,
            P1_Destino=TEXTO_P1[p1], P2_Estres=TEXTO_P2[p2],
            P3_Frecuencia=TEXTO_P3[p3], P4_NivelEstres=estres,
            P5_FinSemana=TEXTO_P5[p5],
        ))

    for _ in range(n_outliers):
        p1 = OPCIONES_LETRA[rng.integers(0, 4)]
        p2 = OPCIONES_LETRA[rng.integers(0, 4)]
        p3 = OPCIONES_LETRA[rng.integers(0, 4)]
        p5 = OPCIONES_LETRA[rng.integers(0, 4)]
        filas.append(dict(
            Edad=int(rng.integers(18, 46)), Genero=str(rng.choice(GENEROS)),
            P1_Destino=TEXTO_P1[p1], P2_Estres=TEXTO_P2[p2],
            P3_Frecuencia=TEXTO_P3[p3], P4_NivelEstres=int(rng.integers(1, 11)),
            P5_FinSemana=TEXTO_P5[p5],
        ))

    df = pd.DataFrame(filas)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Genera datos sinteticos realistas.")
    parser.add_argument("--n",      type=int, default=5000)
    parser.add_argument("--salida", type=str, default="datos_5000_v2.csv")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    df = generar_dataset(args.n, seed=args.seed)
    df.to_csv(args.salida, sep=";", index=False, encoding="utf-8-sig")
    combis = df[["P1_Destino","P2_Estres","P3_Frecuencia","P5_FinSemana"]].drop_duplicates().shape[0]
    print(f"Generadas {len(df)} filas -> {args.salida}")
    print(f"  Arquetipos: {len(ARQUETIPOS)} + {int(args.n * PROP_OUTLIERS)} outliers genuinos")
    print(f"  Combinaciones categoricas unicas: {combis}")


if __name__ == "__main__":
    main()

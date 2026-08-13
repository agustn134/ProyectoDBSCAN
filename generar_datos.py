import pandas as pd
import numpy as np

print("🔄 Cargando datos reales limpios...")
# 1. Cargar el archivo que acabas de limpiar
df_real = pd.read_csv("datos_limpios.csv", sep=";")

# Asegurar tipos de datos numéricos
df_real['Edad'] = df_real['Edad'].astype(int)
df_real['P4_NivelEstres'] = df_real['P4_NivelEstres'].astype(int)

n_real = len(df_real)
print(f" Datos reales cargados correctamente: {n_real} registros.")

# 2. Configuración de generación
total_deseado = 5000
ruido_pct = 0.08  # 8% de ruido
n_ruido = int(total_deseado * ruido_pct)
n_sintetico = total_deseado - n_real - n_ruido

print(f"🧠 Generando {n_sintetico} registros sintéticos balanceados y {n_ruido} registros de ruido...")

# 3. Calcular distribuciones reales para mantener el balance
p_genero = df_real['Genero'].value_counts(normalize=True).to_dict()

# Para Edad y Nivel de Estrés, usamos media y desviación estándar (acotadas a rangos lógicos)
mean_edad, std_edad = df_real['Edad'].mean(), df_real['Edad'].std()
mean_p4, std_p4 = df_real['P4_NivelEstres'].mean(), df_real['P4_NivelEstres'].std()
min_edad, max_edad = int(df_real['Edad'].min()), int(df_real['Edad'].max())

# Función auxiliar para generar categorías respetando las probabilidades reales
def generar_categorias_balanceadas(columna, n):
    probs = df_real[columna].value_counts(normalize=True).to_dict()
    categorias = list(probs.keys())
    probabilidades = list(probs.values())
    return np.random.choice(categorias, size=n, p=probabilidades)

# 4. Generar datos sintéticos balanceados (Vectorizado para velocidad)
data_sintetico = {
    'Edad': np.clip(np.random.normal(mean_edad, std_edad, n_sintetico), min_edad, max_edad).astype(int),
    'Genero': np.random.choice(list(p_genero.keys()), size=n_sintetico, p=list(p_genero.values())),
    'P1_Destino': generar_categorias_balanceadas('P1_Destino', n_sintetico),
    'P2_Estres': generar_categorias_balanceadas('P2_Estres', n_sintetico),
    'P3_Frecuencia': generar_categorias_balanceadas('P3_Frecuencia', n_sintetico),
    'P4_NivelEstres': np.clip(np.random.normal(mean_p4, std_p4, n_sintetico), 1, 10).astype(int),
    'P5_FinSemana': generar_categorias_balanceadas('P5_FinSemana', n_sintetico)
}
df_sintetico = pd.DataFrame(data_sintetico)

# 5. Generar datos de RUIDO (respuestas aleatorias sin correlación para que DBSCAN las detecte)
opciones_p1 = df_real['P1_Destino'].unique().tolist()
opciones_p2 = df_real['P2_Estres'].unique().tolist()
opciones_p3 = df_real['P3_Frecuencia'].unique().tolist()
opciones_p5 = df_real['P5_FinSemana'].unique().tolist()
opciones_genero = df_real['Genero'].unique().tolist()

data_ruido = {
    'Edad': np.random.randint(13, 70, n_ruido),
    'Genero': np.random.choice(opciones_genero, n_ruido),
    'P1_Destino': np.random.choice(opciones_p1, n_ruido),
    'P2_Estres': np.random.choice(opciones_p2, n_ruido),
    'P3_Frecuencia': np.random.choice(opciones_p3, n_ruido),
    'P4_NivelEstres': np.random.randint(1, 11, n_ruido),
    'P5_FinSemana': np.random.choice(opciones_p5, n_ruido)
}
df_ruido = pd.DataFrame(data_ruido)

# 6. Unir todo: Reales + Sintéticos + Ruido
df_final = pd.concat([df_real, df_sintetico, df_ruido], ignore_index=True)

# 7. Mezclar (shuffle) los datos para que los registros reales no estén todos juntos al principio
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# 8. Guardar en formato compatible con tu app
archivo_salida = "datos_5000.csv"
df_final.to_csv(archivo_salida, index=False, sep=";")

print("="*60)
print(f"🎉 ¡ÉXITO TOTAL!")
print(f"Se generó '{archivo_salida}' con {len(df_final)} registros.")
print(f"   - Datos reales: {n_real}")
print(f"   - Datos sintéticos balanceados: {n_sintetico}")
print(f"   - Datos de ruido (atípicos): {n_ruido}")
print("="*60)
print("💡 Este archivo está listo para ser cargado en tu aplicación Streamlit.")
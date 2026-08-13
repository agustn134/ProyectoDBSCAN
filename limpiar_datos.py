import pandas as pd
import numpy as np
import re
import unicodedata

def quitar_acentos(texto):
    """Elimina acentos y caracteres especiales de un texto"""
    if pd.isna(texto):
        return texto
    texto = str(texto)
    # Normalizar y quitar acentos
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    return texto

print("🔄 Limpiando y estandarizando datos de KoboToolbox...")
df = pd.read_csv("datos_reales.csv", sep=';')

print(f" Columnas originales: {len(df.columns)}")
print(f"📋 Filas totales: {len(df)}")

# ============================================================================
# 1. UNIFICAR COLUMNAS DUPLICADAS (KoboToolbox crea columnas al actualizar)
# ============================================================================
df_limpio = pd.DataFrame()

# Edad: tomar de columna 0 o columna 9 (donde esté llena)
df_limpio['Edad'] = df.iloc[:, 0].fillna(df.iloc[:, 9])

# Género: columna 1 o columna 9
if 'Sexo:' in df.columns:
    df_limpio['Genero'] = df['Sexo:'].fillna(df.iloc[:, 9])
else:
    df_limpio['Genero'] = df.iloc[:, 1].fillna(df.iloc[:, 9])

# P1: Destino del gasto (columna 2 o 11)
df_limpio['P1_Destino'] = df.iloc[:, 2].fillna(df.iloc[:, 11])

# P2: Manejo del estrés (columna 3 o 17)
df_limpio['P2_Estres'] = df.iloc[:, 3].fillna(df.iloc[:, 17])

# P3: Frecuencia de salidas (columna 4 o 19)
df_limpio['P3_Frecuencia'] = df.iloc[:, 4].fillna(df.iloc[:, 19])

# P4: Nivel de estrés (columna 6 o 20)
df_limpio['P4_NivelEstres'] = df.iloc[:, 6].fillna(df.iloc[:, 20])

# P5: Fin de semana ideal (columna 5 o 21)
df_limpio['P5_FinSemana'] = df.iloc[:, 5].fillna(df.iloc[:, 21])

# ============================================================================
# 2. LIMPIEZA DE EDAD Y GENERO
# ============================================================================

# Convertir Edad a numérico y eliminar filas sin edad
df_limpio['Edad'] = pd.to_numeric(df_limpio['Edad'], errors='coerce')
df_limpio = df_limpio.dropna(subset=['Edad'])

# Asegurar que Edad sea entero
df_limpio['Edad'] = df_limpio['Edad'].astype(int)

# Limpiar Género: quitar acentos y estandarizar
df_limpio['Genero'] = df_limpio['Genero'].apply(quitar_acentos)
df_limpio['Genero'] = df_limpio['Genero'].str.strip()
# Estandarizar a "Hombre" o "Mujer"
df_limpio['Genero'] = df_limpio['Genero'].apply(lambda x: 'Hombre' if 'hombre' in x.lower() else ('Mujer' if 'mujer' in x.lower() else x))

# ============================================================================
# 3. FUNCIÓN DE ESTANDARIZACIÓN (mapea respuestas viejas → formato nuevo)
# ============================================================================
def estandarizar_respuesta(texto, tipo_pregunta):
    """
    Limpia emojis, quita acentos y estandariza al formato final breve.
    """
    if pd.isna(texto):
        return np.nan
    
    texto = str(texto).strip()
    
    # Eliminar emojis y caracteres especiales
    texto = re.sub(r'[^\w\s\)\(\-\.\,]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    # Quitar acentos
    texto = quitar_acentos(texto)
    
    # ========================================================================
    # MAPEO DE RESPUESTAS ANTIGUAS → FORMATO NUEVO
    # ========================================================================
    
    if tipo_pregunta == 'P1':  # Destino del gasto
        if any(x in texto.lower() for x in ['fiesta', 'bebida', 'noche', 'salir', 'beber', 'amigos']):
            return "A) Salir a beber con amigos y fiestas."
        elif any(x in texto.lower() for x in ['gimnasio', 'salud', 'comida', 'naturaleza', 'terapia', 'ejercicio']):
            return "B) Gimnasio, comida y salud."
        elif any(x in texto.lower() for x in ['ahorro', 'invertir', 'inversion', 'meta', 'futuro', 'ahorrando']):
            return "C) Ahorrando o invirtiendolo."
        elif any(x in texto.lower() for x in ['viaje', 'concierto', 'curso', 'taller', 'aventura', 'deporte', 'nueva actividad']):
            return "D) Viajes, conciertos, cursos."
        else:
            return texto
    
    elif tipo_pregunta == 'P2':  # Manejo del estrés
        if any(x in texto.lower() for x in ['salir', 'desconectar', 'beber', 'socializar']):
            return "A) Salir a desconectarme, beber y socializar."
        elif any(x in texto.lower() for x in ['ejercicio', 'meditar', 'actividad fisica']):
            return "B) Hacer ejercicio y meditar."
        elif any(x in texto.lower() for x in ['descansar', 'casa', 'planeando', 'tiempo', 'gastos']):
            return "C) Descansar en casa, planeando mi tiempo y gastos."
        elif any(x in texto.lower() for x in ['nueva', 'actividad', 'viajar', 'buscar']):
            return "D) Buscar nuevas actividades o viajando."
        else:
            return texto
    
    elif tipo_pregunta == 'P3':  # Frecuencia de salidas
        if '4' in texto and ('mas' in texto.lower() or 'más' in texto.lower()):
            return "A) 4 o mas veces."
        elif '1' in texto and '2' in texto:
            return "B) 1 o 2 veces."
        elif any(x in texto.lower() for x in ['casi', 'nunca', 'no']):
            return "C) Casi nunca."
        elif any(x in texto.lower() for x in ['evento', 'importante', 'concierto']):
            return "D) Solo cuando hay un evento importante."
        else:
            return texto
    
    elif tipo_pregunta == 'P5':  # Fin de semana ideal
        if any(x in texto.lower() for x in ['improvisado', 'amigos', 'musica']):
            return "A) Improvisado, con amigos y musica."
        elif any(x in texto.lower() for x in ['activo', 'deporte', 'haciendo', 'comiendo']):
            return "B) Activo, haciendo deporte y comiendo."
        elif any(x in texto.lower() for x in ['tranquilo', 'casa', 'preparando', 'semana']):
            return "C) Tranquilo, en casa y preparando la semana."
        else:
            return texto
    
    return texto

# ============================================================================
# 4. APLICAR ESTANDARIZACIÓN A TODAS LAS COLUMNAS
# ============================================================================
print("\n🧹 Estandarizando respuestas...")
df_limpio['P1_Destino'] = df_limpio['P1_Destino'].apply(lambda x: estandarizar_respuesta(x, 'P1'))
df_limpio['P2_Estres'] = df_limpio['P2_Estres'].apply(lambda x: estandarizar_respuesta(x, 'P2'))
df_limpio['P3_Frecuencia'] = df_limpio['P3_Frecuencia'].apply(lambda x: estandarizar_respuesta(x, 'P3'))
df_limpio['P5_FinSemana'] = df_limpio['P5_FinSemana'].apply(lambda x: estandarizar_respuesta(x, 'P5'))

# ============================================================================
# 5. LIMPIEZA FINAL
# ============================================================================

# Eliminar filas con respuestas incompletas
columnas_clave = ['P1_Destino', 'P2_Estres', 'P3_Frecuencia', 'P4_NivelEstres', 'P5_FinSemana']
df_limpio = df_limpio.dropna(subset=columnas_clave)

# Convertir P4 a entero
df_limpio['P4_NivelEstres'] = pd.to_numeric(df_limpio['P4_NivelEstres'], errors='coerce')
df_limpio = df_limpio.dropna(subset=['P4_NivelEstres'])
df_limpio['P4_NivelEstres'] = df_limpio['P4_NivelEstres'].astype(int)

# ============================================================================
# 6. ORDENAR COLUMNAS (SIN ACENTOS)
# ============================================================================
df_limpio = df_limpio[[
    'Edad',
    'Genero',
    'P1_Destino',
    'P2_Estres',
    'P3_Frecuencia',
    'P4_NivelEstres',
    'P5_FinSemana'
]]

# ============================================================================
# 7. GUARDAR RESULTADOS
# ============================================================================
archivo_excel = "datos_limpios.xlsx"
df_limpio.to_excel(archivo_excel, index=False, engine='openpyxl')
print(f"\n✅ Archivo Excel guardado: {archivo_excel}")

archivo_csv = "datos_limpios.csv"
df_limpio.to_csv(archivo_csv, index=False, sep=';')
print(f"✅ Archivo CSV guardado: {archivo_csv}")

# ============================================================================
# 8. ESTADÍSTICAS
# ============================================================================
print("\n" + "="*60)
print(" RESUMEN DE DATOS LIMPIOS")
print("="*60)
print(f"✓ Total de respuestas válidas: {len(df_limpio)}")
print(f"✓ Edad promedio: {df_limpio['Edad'].mean():.1f} años")
print(f"✓ Edad mínima: {df_limpio['Edad'].min()} años")
print(f"✓ Edad máxima: {df_limpio['Edad'].max()} años")
print(f"✓ Nivel de estrés promedio: {df_limpio['P4_NivelEstres'].mean():.1f}/10")
print(f"\n✓ Distribución por género:")
print(df_limpio['Genero'].value_counts())
print("="*60)

# Mostrar ejemplos de respuestas estandarizadas
print("\n📝 EJEMPLOS DE RESPUESTAS ESTANDARIZADAS:")
print("="*60)
for i in range(min(3, len(df_limpio))):
    print(f"\nFila {i+1}:")
    print(f"  Edad: {df_limpio.iloc[i]['Edad']}")
    print(f"  Genero: {df_limpio.iloc[i]['Genero']}")
    print(f"  P1: {df_limpio.iloc[i]['P1_Destino']}")
    print(f"  P2: {df_limpio.iloc[i]['P2_Estres']}")
    print(f"  P3: {df_limpio.iloc[i]['P3_Frecuencia']}")
    print(f"  P5: {df_limpio.iloc[i]['P5_FinSemana']}")

print("\n🎉 ¡Listo! Tus datos están completamente estandarizados.")
print("   Usa 'datos_limpios.csv' en tu aplicación Streamlit.")
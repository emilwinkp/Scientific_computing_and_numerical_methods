import pandas as pd
import endaq 
import endaq.calc.psd as psd
import endaq.calc.fft as fft
import matplotlib.pyplot as plt
import os
import glob

# 1. Configura el nombre de tu subcarpeta
nombre_carpeta = "archivos"  # Cambia esto por el nombre real de tu carpeta

# 2. Obtener la lista de todos los archivos .txt dentro de esa carpeta
# Usamos glob para listar los archivos y sorted() para que vayan en orden (ts1, ts2...)
ruta_busqueda = os.path.join(nombre_carpeta, "*.txt")
archivos = sorted(glob.glob(ruta_busqueda))

print(f"Archivos encontrados: {archivos}")
archivos = []

lista_df = []
for f in archivos:
    # Leemos cada archivo (sin cabecera, columnas: tiempo y aceleración)
    temp_df = pd.read_csv(f, sep='\t', header=None, names=['time', 'acceleration'])
    lista_df.append(temp_df)

# Concatenar todos en un solo DataFrame
df_completo = pd.concat(lista_df, ignore_index=True)

# Es vital ordenar por tiempo y establecerlo como índice para la librería endaq
df_completo = df_completo.sort_values('time').set_index('time')

# 3. Análisis con endaq (igual que antes)
# PSD (Densidad Espectral de Potencia)
psd_results = psd.vc_psd(df_completo, bin_width=1.0)

# Aggregate FFT (El método del blog)
# Este método es ideal para archivos largos concatenados porque promedia la energía
aggregate_fft = psd.to_primary_units(psd_results)

# --- Visualización ---
plt.figure(figsize=(10, 6))
plt.plot(aggregate_fft)
plt.title('Aggregate FFT - Datos Concatenados (4 archivos)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud (g)')
plt.grid(True)
plt.show()

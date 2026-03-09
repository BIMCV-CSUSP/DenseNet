import pandas as pd
import nibabel as nib
import os

# 1. Ruta a tu CSV
csv_path = '/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/code/total_segmentator/updated_patients_per_discs.csv'

print("Cargando rutas del CSV...")
df = pd.read_csv(csv_path)

# Limpiar las que no existen por si acaso
rutas_validas = [path for path in df['disc_path'].dropna() if os.path.exists(path)]
print(f"Analizando {len(rutas_validas)} imágenes. Esto tomará unos segundos...")

shapes = []

# 2. Leer las dimensiones de cada imagen
for path in rutas_validas:
    try:
        # nibabel lee la cabecera rápido sin cargar toda la matriz 3D
        img = nib.load(path)
        shapes.append(img.shape)
    except Exception as e:
        print(f"Error leyendo {path}: {e}")

# 3. Crear un DataFrame con los resultados
shapes_df = pd.DataFrame(shapes, columns=['Eje X (Ancho)', 'Eje Y (Alto)', 'Eje Z (Profundidad)'])

# 4. Imprimir las estadísticas clave
print("\n" + "="*40)
print("ESTADÍSTICAS DE TAMAÑO DE TUS IMÁGENES")
print("="*40)
print(shapes_df.describe().round(1))

print("\n" + "="*40)
print("EL TAMAÑO 'MEDIANO' IDEAL ES APROXIMADAMENTE:")
print(shapes_df.median().astype(int))
print("="*40)
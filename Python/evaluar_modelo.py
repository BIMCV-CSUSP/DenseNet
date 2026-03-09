# import logging
# import os
# import sys
# import numpy as np
# import torch
# import monai
# from monai.data import DataLoader
# from monai.transforms import Compose, LoadImaged, Orientationd, Spacingd, ScaleIntensityd, Resized
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt
# import nibabel as nib

# def main():
#     # ==========================================
#     # 1. CONFIGURATION (CHANGE THIS!)
#     # ==========================================
#     # Find the .pth file generated in your 'runs' folder
#     # Example: "runs/Feb09_11-00-00_server/best_metric_model_classification3d_dict.pth"
#     MODEL_PATH = "/mnt/datalake/openmind/MedP-Midas/sgonzalez/DL/DL/runs/Jan07_11-06-27_tartaglia01/best_metric_model_classification3d_dict.pth" 
    
#     # Original data file
#     CSV_PATH = '/mnt/datalake/openmind/MedP-Midas/sgonzalez/DL/DL/data/updated_patients_per_discs.csv'
    
#     device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # ==========================================
#     # 2. PREPARE DATA (Identical to your training)
#     # ==========================================
#     # It is CRITICAL to use the same random_state=42 so the validation set remains the same
#     data = pd.read_csv(CSV_PATH)
#     data['Pfirrmann'] = pd.to_numeric(data['Pfirrmann'], errors='coerce')
#     data['Pfirrmann'] = data['Pfirrmann']
#     data = data.dropna(subset=['Pfirrmann'])
#     data['Pfirrmann'] = data['Pfirrmann'].astype(int)

#     # Filter broken files (same as before)
#     data = data[data['disc_path'].apply(os.path.exists)]

#     unique_patients = data['patient_id'].unique()
#     # Same random_state as in training to recover the same patients
#     _, val_patients = train_test_split(unique_patients, test_size=0.1, random_state=42)
    
#     val_df = data[data['patient_id'].isin(val_patients)]
#     val_files = [{"img": img, "label": label} for img, label in zip(val_df['disc_path'].values, val_df['Pfirrmann'].values)]

#     print(f"Validating with {len(val_files)} images.")

#     # VALIDATION transforms (no rotations, only resize and norm)
#     val_transforms = Compose([
#         LoadImaged(keys=["img"], ensure_channel_first=True),
#         Orientationd(keys=["img"], axcodes="RAS"),
#         Spacingd(keys=["img"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
#         ScaleIntensityd(keys=["img"]),
#         Resized(keys=["img"], spatial_size=(96, 64, 32)),
#     ])

#     val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
#     val_loader = DataLoader(val_ds, batch_size=4, num_workers=0, pin_memory=torch.cuda.is_available())

#     # ==========================================
#     # 3. LOAD TRAINED MODEL
#     # ==========================================
#     model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=5).to(device)
    
#     print(f"Loading weights from: {MODEL_PATH}")
#     try:
#         model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     except FileNotFoundError:
#         print("ERROR: .pth file not found. Check the MODEL_PATH variable at the start of the script.")
#         return

#     model.eval()

#     # ==========================================
#     # 4. GENERATE PREDICTIONS (FIXED)
#     # ==========================================
#     y_pred_all = []
#     y_true_all = []

#     with torch.no_grad():
#         for val_data in val_loader:
#             val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
#             outputs = model(val_images)
            
#             # Shift predictions from 0-4 back to 1-5
#             preds = outputs.argmax(dim=1).cpu().numpy() + 1
            
#             # Ensure labels are also 1-5
#             # If you subtracted 1 from val_labels earlier in the script, add +1 here too
#             labels = val_labels.cpu().numpy() 
            
#             y_pred_all.extend(preds)
#             y_true_all.extend(labels)

#     # ==========================================
#     # 5. DRAW MATRIX (FIXED)
#     # ==========================================
#     # Now we look for labels 1 through 5
#     cm = confusion_matrix(y_true_all, y_pred_all, labels=[1, 2, 3, 4, 5]) 

#     class_names = ["G1", "G2", "G3", "G4", "G5"]
#     fig, ax = plt.subplots(figsize=(10, 8))
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
#     disp.plot(cmap='Blues', ax=ax, values_format='d')
    
#     plt.title('Confusion Matrix (Validation Set)')
#     output_file = "/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/code/DL/matriz_confusion_final.png"
#     plt.savefig(output_file)
#     plt.close()
    
#     print(f"Done! Image saved at: {os.path.abspath(output_file)}")

# if __name__ == "__main__":
#     main()



import os
import sys
import torch
import monai
from monai.data import DataLoader, CacheDataset
from monai.transforms import Compose, LoadImaged, Orientationd, Spacingd, ScaleIntensityd, Resized
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main():
    # ==========================================
    # 1. CONFIGURACIÓN (¡CAMBIA LA RUTA DE LA CARPETA!)
    # ==========================================
    # Pon aquí el nombre de la carpeta 'runs' que se acaba de generar hoy
    # Ejemplo: "runs/Feb23_15-00-00"
    RUN_FOLDER = "/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/code/total_segmentator/runs/Feb20_09-12-29" 
    
    MODEL_PATH = os.path.join(RUN_FOLDER, "best_metric_model.pt")
    CSV_PATH = '/mnt/datalake/openmind/MedP-Midas/sgonzalez/DL/DL/data/updated_patients_per_discs.csv'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*30)
    print(f"USING DEVICE: {device}")
    print("="*30 + "\n")

    # ==========================================
    # 2. PREPARAR DATOS EXACTAMENTE IGUAL QUE EN EL TRAIN
    # ==========================================
    data = pd.read_csv(CSV_PATH)
    data['Pfirrmann'] = pd.to_numeric(data['Pfirrmann'], errors='coerce')
    data = data.dropna(subset=['Pfirrmann'])
    data['Pfirrmann'] = data['Pfirrmann'].astype(int) - 1

    valid_paths = [os.path.exists(p) for p in data['disc_path']]
    data = data[valid_paths]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    _, val_idx = next(gss.split(data, groups=data['patient_id']))
    val_df = data.iloc[val_idx]

    val_files = [{"img": img, "label": label} for img, label in zip(val_df['disc_path'].values, val_df['Pfirrmann'].values)]
    print(f"Validando con {len(val_files)} imágenes (Mismo set que en el entrenamiento).")

    val_transforms = Compose([
        LoadImaged(keys=["img"], ensure_channel_first=True),
        Orientationd(keys=["img"], axcodes="RAS"),
        Spacingd(keys=["img"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=(96, 64, 32)),
    ])

    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=0, pin_memory=torch.cuda.is_available())

    # ==========================================
    # 3. CARGAR EL MODELO
    # ==========================================
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=5).to(device)
    
    print(f"Cargando pesos desde: {MODEL_PATH}")
    try:
        # weights_only=True quita el warning de seguridad de PyTorch
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"ERROR: No encuentro el archivo {MODEL_PATH}. Revisa la variable RUN_FOLDER.")
        return

    model.eval()

    # ==========================================
    # 4. INFERENCIA CON BARRA DE PROGRESO
    # ==========================================
    y_pred_all = []
    y_true_all = []

    print("\nIniciando inferencia (esto puede tardar un poco si vas por CPU)...")
    total_batches = len(val_loader)
    
    with torch.no_grad():
        for step, val_data in enumerate(val_loader, 1):
            val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
            outputs = model(val_images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels = val_labels.cpu().numpy()
            
            y_pred_all.extend(preds)
            y_true_all.extend(labels)
            
            # Print de progreso para que sepas que no está colgado
            print(f"Procesando batch {step}/{total_batches}...", end="\r")

    # ==========================================
    # 5. DIBUJAR MATRIZ
    # ==========================================
    print("\n\nCreando gráfica de la matriz de confusión...")
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1, 2, 3, 4])
    class_names = ["G1", "G2", "G3", "G4", "G5"]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    
    plt.title('Matriz de Confusión (Set de Validación Sin Fuga)')
    output_file = os.path.join(RUN_FOLDER, "matriz_confusion_final.png")
    plt.savefig(output_file)
    plt.close()
    
    print(f"¡Listo! Imagen guardada en: {output_file}")

if __name__ == "__main__":
    main()
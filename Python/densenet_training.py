# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########## WE USED THIS ENVIRONMENT /mnt/datalake/FISABIO_datalake/pituitaria/venv_311 
import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityd, Orientationd, Spacingd, RandRotated
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import nibabel as nib

# NEW: Import matplotlib for plotting
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit

# NUEVO: Importar métricas de Scikit-Learn
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def main():
    # --- ARGS CONFIGURATION ---
    parser = argparse.ArgumentParser(description="DenseNet Training with MONAI")
    parser.add_argument("--csv", type=str, default='/mnt/datalake/openmind/MedP-Midas/sgonzalez/DL/DL/data/updated_patients_per_discs.csv', help="Path to the input CSV")
    parser.add_argument("--img_col", type=str, default='disc_path', help="Column name for image paths")
    parser.add_argument("--label_col", type=str, default='Pfirrmann', help="Column name for labels")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    args = parser.parse_args()

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # --- DEVICE CHECK ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*30)
    print(f"USING DEVICE: {device}")
    if device.type == 'cuda':
        print(f"GPU MODEL: {torch.cuda.get_device_name(0)}")
    print("="*30 + "\n")

    # read csv using args
    data = pd.read_csv(args.csv)

    data[args.label_col] = pd.to_numeric(data[args.label_col], errors='coerce')
    data = data.dropna(subset=[args.label_col])
    data[args.label_col] = data[args.label_col].astype(int) - 1  # Shift 1-5 to 0-4
    
    # Check file existence and clean dataframe
    missing_files_count = 0
    valid_paths = []
    for path in data[args.img_col].tolist():
        if os.path.exists(path):
            valid_paths.append(True)
        else:
            valid_paths.append(False)
            missing_files_count += 1
    
    data = data[valid_paths]
    print(f"Total missing files removed: {missing_files_count}")

    # # Stratify by label to ensure all grades (0-4) are in both sets
    # train_df, val_df = train_test_split(
    #     data, 
    #     test_size=0.1, 
    #     random_state=42, 
    #     stratify=data[args.label_col]
    # )
    




    # ... (después de limpiar tu dataframe) ...

    # NUEVO: Split por grupos (Pacientes) para EVITAR DATA LEAKAGE
    print("Dividiendo datos a nivel de PACIENTE (Evitando Data Leakage)...")
    
    # Asegúrate de que tienes la columna 'patient_id' en tu dataframe cargado
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    
    # Esto garantiza que todos los discos de un paciente van juntos al mismo set
    train_idx, val_idx = next(gss.split(data, groups=data['patient_id']))
    
    train_df = data.iloc[train_idx]
    val_df = data.iloc[val_idx]

    print(f"Pacientes en Train: {train_df['patient_id'].nunique()} | Discos: {len(train_df)}")
    print(f"Pacientes en Val:   {val_df['patient_id'].nunique()} | Discos: {len(val_df)}")


    train_files = [{"img": img, "label": label} for img, label in zip(train_df[args.img_col].values, train_df[args.label_col].values)]
    random.shuffle(train_files)
    train_files = sorted(train_files, key=lambda x: x['label'], reverse=True)
    val_files = [{"img": img, "label": label} for img, label in zip(val_df[args.img_col].values, val_df[args.label_col].values)]

    # Define transforms
    train_transforms = Compose([
        LoadImaged(keys=["img"], ensure_channel_first=True),
        Orientationd(keys=["img"], axcodes="RAS"),
        Spacingd(keys=["img"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        ScaleIntensityd(keys=["img"]),
        RandRotated(
            keys=["img"],
            prob=0.2,
            range_x=0.087, range_y=0.087, range_z=0.087,
            mode="bilinear",
            padding_mode="border"
        ),
        Resized(keys=["img"], spatial_size=(96, 64, 32)),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["img"], ensure_channel_first=True),
        Orientationd(keys=["img"], axcodes="RAS"),
        Spacingd(keys=["img"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=(96, 64, 32)),
    ])
    
    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=5)])
    post_label = Compose([AsDiscrete(to_onehot=5)])

    # Dataloaders
    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=0, pin_memory=torch.cuda.is_available())

    # Create Model
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=5,dropout_prob=0.4).to(device)
    print("Calculando pesos de clases para equilibrar el entrenamiento...")
    
    # Contamos cuántas imágenes hay de cada clase en el set de ENTRENAMIENTO
    class_counts = train_df[args.label_col].value_counts().sort_index().values
    print(f"Distribución en Train (G1 a G5): {class_counts}")
    
    # Convertimos a tensor
    class_counts_tensor = torch.tensor(class_counts, dtype=torch.float32)
    
    # El peso es inverso a la frecuencia (menos imágenes = mayor peso)
    weights = 1.0 / class_counts_tensor
    
    # Normalizamos para mantener los gradientes estables
    weights = weights / weights.sum() * 5.0
    class_weights = weights.to(device)
    
    print(f"Pesos aplicados (G1 a G5): {class_weights.cpu().numpy().round(2)}")

    # Usamos los pesos en la función de pérdida
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    # loss_function = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3) # <-- NUEVO: weight_decay
    auc_metric = ROCAUCMetric()

    # Training history
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    train_loss_values = []
    train_acc_values = []
    val_loss_values = []
    val_acc_values = []
    val_epochs = []

    # Initialize SummaryWriter (it creates the 'runs/TIMESTAMP' folder)
    log_dir = os.path.join("runs", datetime.now().strftime("%b%d_%H-%M-%S"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Logging to: {log_dir}")
    
    for epoch in range(args.epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.epochs}")
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        total_train_samples = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            preds = outputs.argmax(dim=1)
            epoch_correct += torch.eq(preds, labels).sum().item()
            total_train_samples += len(labels)
            
            epoch_len = len(train_ds) // train_loader.batch_size
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        
        # Guardar Train Loss
        epoch_loss /= step
        train_loss_values.append(epoch_loss)
        
        # NUEVO: Guardar Train Acc
        train_acc = epoch_correct / total_train_samples
        train_acc_values.append(train_acc)
        
        print(f"Train -> Loss: {epoch_loss:.4f} | Acc: {train_acc:.4f}")
        writer.add_scalar("train_accuracy", train_acc, epoch + 1)

        ##VALIDACION

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0 # NUEVO
            val_step = 0 # NUEVO
            
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                
                for val_data in val_loader:
                    val_step += 1
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                    
                    outputs = model(val_images)
                    
                    # NUEVO: Calcular Validation Loss
                    v_loss = loss_function(outputs, val_labels)
                    val_loss += v_loss.item()
                    
                    y_pred = torch.cat([y_pred, outputs], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                # Guardar Validation Loss
                val_loss /= val_step
                val_loss_values.append(val_loss)
                writer.add_scalar("val_loss", val_loss, epoch + 1)

                # Calcular Validation Accuracy
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                
                # Calcular AUC
                y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                auc_result = auc_metric.aggregate()
                auc_metric.reset()

                # NUEVO: Calcular Precision, Recall, Specificity y F1-Score (Macro)
                y_true_np = y.cpu().numpy()
                y_pred_np = y_pred.argmax(dim=1).cpu().numpy()
                
                # Zero_division=0 evita warnings al principio cuando no predice alguna clase
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true_np, y_pred_np, average='macro', zero_division=0
                )
                
                # Para la Especificidad (True Negative Rate), usamos la matriz de confusión
                cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1, 2, 3, 4])
                spec_list = []
                for i in range(5):
                    tn = cm.sum() - (cm[i,:].sum() + cm[:,i].sum() - cm[i,i])
                    fp = cm[:,i].sum() - cm[i,i]
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                    spec_list.append(spec)
                specificity = np.mean(spec_list)

                # Imprimir resumen de métricas
                print(f"Val   -> Loss: {val_loss:.4f} | Acc: {acc_metric:.4f} | AUC: {auc_result:.4f}")
                print(f"         Prec: {precision:.4f} | Sens(Rec): {recall:.4f} | Spec: {specificity:.4f} | F1: {f1:.4f}")

                # Tracking y Guardado del mejor modelo
                val_acc_values.append(acc_metric)
                val_epochs.append(epoch + 1)

                if acc_metric > best_metric:
                    best_metric = acc_metric
                    best_metric_epoch = epoch + 1
                    best_model_path = os.path.join(log_dir, "best_metric_model.pt")
                    torch.save(model.state_dict(), best_model_path)
                    print(f"*** Nuevo Mejor Modelo Guardado! (Acc: {best_metric:.4f}) ***")

                writer.add_scalar("val_accuracy", acc_metric, epoch + 1)

                # =======================
                # NUEVO: PLOT CON 4 LÍNEAS
                # =======================
                fig, ax1 = plt.subplots(figsize=(12, 7))

                # Eje Y Izquierdo (LOSS - Azul/Cian)
                ax1.plot(range(1, epoch + 2), train_loss_values, color='blue', linestyle='-', linewidth=2, label='Train Loss')
                ax1.plot(val_epochs, val_loss_values, color='cyan', linestyle='--', marker='s', label='Val Loss')
                ax1.set_xlabel('Epochs', fontweight='bold')
                ax1.set_ylabel('Loss (CrossEntropy)', color='blue', fontweight='bold')
                ax1.tick_params(axis='y', labelcolor='blue')
                ax1.grid(True, alpha=0.3)

                # Eje Y Derecho (ACCURACY - Rojo/Naranja)
                ax2 = ax1.twinx()
                ax2.plot(range(1, epoch + 2), train_acc_values, color='red', linestyle='-', linewidth=2, label='Train Acc')
                ax2.plot(val_epochs, val_acc_values, color='orange', linestyle='--', marker='o', label='Val Acc')
                ax2.set_ylabel('Accuracy', color='red', fontweight='bold')
                ax2.tick_params(axis='y', labelcolor='red')
                ax2.set_ylim(0, 1.05)

                # Combinar Leyendas
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
                
                plt.title(f'Progreso de Entrenamiento (Época {epoch + 1})', fontsize=14, fontweight='bold')

                # Marca del Mejor Modelo
                if best_metric_epoch != -1:
                    ax2.scatter([best_metric_epoch], [best_metric], color='gold', s=150, marker='*', zorder=5, edgecolors='black')

                plt.savefig(os.path.join(log_dir, "progress.png"), bbox_inches='tight')
                plt.close()

    print(f"Train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

if __name__ == "__main__":
    main()
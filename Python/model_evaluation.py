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
    # 1. CONFIGURATION (CHANGE THE FOLDER PATH!)
    # ==========================================
    # Set the name of the 'runs' folder generated today
    # Example: "runs/Feb23_15-00-00"
    RUN_FOLDER = "/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/code/total_segmentator/runs/Feb20_09-12-29" 
    
    MODEL_PATH = os.path.join(RUN_FOLDER, "best_metric_model.pt")
    CSV_PATH = '/mnt/datalake/openmind/MedP-Midas/sgonzalez/DL/DL/data/updated_patients_per_discs.csv'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*30)
    print(f"USING DEVICE: {device}")
    print("="*30 + "\n")

    # ==========================================
    # 2. PREPARE DATA EXACTLY AS IN TRAINING
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
    print(f"Validating with {len(val_files)} images (Same set as in training).")

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
    # 3. LOAD THE MODEL
    # ==========================================
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=5).to(device)
    
    print(f"Loading weights from: {MODEL_PATH}")
    try:
        # weights_only=True suppresses the PyTorch security warning
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"ERROR: Could not find file {MODEL_PATH}. Check the RUN_FOLDER variable.")
        return

    model.eval()

    # ==========================================
    # 4. INFERENCE WITH PROGRESS TRACKING
    # ==========================================
    y_pred_all = []
    y_true_all = []

    print("\nStarting inference (this might take a while if using CPU)...")
    total_batches = len(val_loader)
    
    with torch.no_grad():
        for step, val_data in enumerate(val_loader, 1):
            val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
            outputs = model(val_images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels = val_labels.cpu().numpy()
            
            y_pred_all.extend(preds)
            y_true_all.extend(labels)
            
            # Progress update
            print(f"Processing batch {step}/{total_batches}...", end="\r")

    # ==========================================
    # 5. PLOT CONFUSION MATRIX
    # ==========================================
    print("\n\nGenerating confusion matrix plot...")
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1, 2, 3, 4])
    class_names = ["G1", "G2", "G3", "G4", "G5"]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    
    plt.title('Confusion Matrix (Leak-free Validation Set)')
    output_file = os.path.join(RUN_FOLDER, "final_confusion_matrix.png")
    plt.savefig(output_file)
    plt.close()
    
    print(f"Done! Image saved at: {output_file}")

if __name__ == "__main__":
    main()
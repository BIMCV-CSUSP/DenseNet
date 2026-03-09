# 3D DenseNet for Pfirrmann Grading Classification

This repository contains a deep learning pipeline for the automated classification of intervertebral disc degeneration using the Pfirrmann grading system (Grades 1 to 5).

The core of this project consists of two main scripts:

- densenet_training.py: Trains a 3D DenseNet121 model on MRI NIfTI volumes.

- model_evaluation.py: Evaluates the trained model on the validation set and generates a detailed Confusion Matrix.

##  Key Features (densenet_training.py)

This script is highly optimized for medical imaging data and includes several advanced machine learning techniques to ensure robust, real-world performance:

* **3D DenseNet Architecture:** Utilizes MONAI's `DenseNet121` configured for 3D spatial dimensions to capture volumetric disc features.
* **Data Leakage Prevention:** Uses Scikit-Learn's `GroupShuffleSplit` to split the dataset by `patient_id`. This ensures that multiple discs from the same patient are kept in the same split (Train or Validation), preventing the model from "memorizing" patient-specific anatomy.
* **Dynamic Class Weighting:** Automatically calculates and applies weights to the `CrossEntropyLoss` function based on the training set distribution. This directly addresses class imbalance (e.g., rare Grade 1 and Grade 5 cases vs. common Grade 3 and 4 cases).
* **Regularization against Overfitting:** Implements a 40% Dropout (`dropout_prob=0.4`) inside the DenseNet and Weight Decay (`weight_decay=1e-3`) in the Adam optimizer to force the model to generalize rather than memorize.
* **High-Speed Caching:** Uses MONAI's `CacheDataset` to load and apply deterministic transforms to the RAM upfront, drastically reducing disk I/O bottlenecks and speeding up training epochs.
* **Comprehensive Metrics Tracking:** Calculates and logs Validation Loss, Accuracy, AUC, Precision, Sensitivity (Recall), Specificity, and Macro F1-Score.
* **Automated Visualization:** Generates a dual-axis `progress.png` chart every validation step, tracking Train/Val Loss and Train/Val Accuracy over time.

## Model evaluation (model_evaluation.py-)
Once a model is trained, this script is used to securely load the best weights and perform a detailed evaluation on the validation cohort.

- Confusion Matrix Generation: Automatically computes and plots a multi-class confusion matrix using Scikit-Learn and Matplotlib, saving it directly to the experiment's run folder for easy clinical analysis.
- 
---

## 📂 Input Data Format

The model requires a **CSV file** and a directory of **3D NIfTI (`.nii` or `.nii.gz`) images**.

### The CSV File

By default, the script expects a CSV file containing at least the following columns:

* `disc_path`: The absolute path to the cropped 3D NIfTI file of the intervertebral disc.
* `Pfirrmann`: The clinical ground truth label (integer from 1 to 5). *Note: The script automatically shifts these to 0-4 for PyTorch compatibility.*
* `patient_id`: A unique identifier for the patient. Used to safely split the data without leakage.

**Example CSV structure:**

```csv
patient_id,disc_path,Pfirrmann
Pat_001,/data/images/Pat_001_L1_L2.nii.gz,2
Pat_001,/data/images/Pat_001_L2_L3.nii.gz,3
Pat_002,/data/images/Pat_002_L4_L5.nii.gz,5

```

### The 3D Images

The images should be 3D MRI crops of individual intervertebral discs. The script automatically applies the following MONAI preprocessing pipeline to standardize them:

1. **Orientation:** Reoriented to `RAS` coordinate system.
2. **Spacing:** Resampled to an isotropic voxel size of `1.0 x 1.0 x 1.0 mm`.
3. **Intensity:** Scaled/Normalized using `ScaleIntensityd`.
4. **Augmentation (Train only):** Random 3D rotations (`RandRotated`) with a 20% probability.
5. **Resizing:** Standardized to a spatial size of `96 x 64 x 32` voxels to fit the DenseNet architecture and optimize GPU memory.

---

## 🛠️ Usage

### Installation

Ensure you have the required libraries installed in your Python environment:

```bash
pip install torch monai numpy pandas scikit-learn nibabel matplotlib tensorboard

```

### Running the Training Script

You can run the script via the command line. It accepts several arguments to customize the run:

```bash
python densenet_training.py \
    --csv "/path/to/your/dataset.csv" \
    --img_col "disc_path" \
    --label_col "Pfirrmann" \
    --epochs 150

```

**Arguments:**

* `--csv`: Path to your input CSV file.
* `--img_col`: Name of the column containing image paths (default: `disc_path`).
* `--label_col`: Name of the column containing the target labels (default: `Pfirrmann`).
* `--epochs`: Total number of training epochs (default: `150`).

---

## 📊 Outputs

During and after training, the script will create a `runs/` directory with a timestamped folder for the current experiment (e.g., `runs/Jan01_12-00-00/`). Inside this folder, you will find:

1. **`best_metric_model.pt`**: The saved PyTorch model weights from the epoch that achieved the highest Validation Accuracy.
2. **`progress.png`**: An automatically updated matplotlib chart showing:
* Train Loss (Blue solid line) vs. Val Loss (Cyan dashed line).
* Train Accuracy (Red solid line) vs. Val Accuracy (Orange dashed line).
* A gold star marking the epoch where the best model was saved.


3. **TensorBoard Logs**: Event files tracking loss, accuracy, and AUC. You can visualize them by running:
```bash
tensorboard --logdir=runs/

```

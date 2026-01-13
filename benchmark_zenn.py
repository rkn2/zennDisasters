
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

def benchmark():
    # Load Data
    y_true_cont = pd.read_csv("real_tornado_targets.csv", header=None).values.flatten()
    y_pred_cont = np.loadtxt("real_zenn_predictions.txt")
    
    # Original target was 0, 1, 2 (Undamaged, Low, Significant)
    # Our data: Normalized 0-1.
    # We need to map back to classes to compare F1.
    # Assuming original normalization was / max(damage), where max=4 or 5?
    # Let's inspect unique values in y_true to deterime discrete levels.
    
    unique_vals = np.unique(y_true_cont)
    print(f"Unique target values (normalized): {unique_vals}")
    
    # Simple Thresholding Logic
    # If 3 classes: [0, 0.5, 1.0] -> Cutoffs at 0.25, 0.75?
    # If 4 classes: [0, 0.33, 0.66, 1.0] -> Cutoffs 0.16, 0.5, 0.83?
    
    # Let's cluster predictions or just round to nearest unique target value
    # to "classify" the regression output.
    
    y_true_class = []
    y_pred_class = []
    
    # Nearest Neighbor mapping to valid classes
    for val in y_pred_cont:
        idx = (np.abs(unique_vals - val)).argmin()
        y_pred_class.append(unique_vals[idx])
        
    for val in y_true_cont:
        # Should match exactly but float errors possible
        idx = (np.abs(unique_vals - val)).argmin()
        y_true_class.append(unique_vals[idx])
        
    y_true_class = np.array(y_true_class).round(2).astype(str)
    y_pred_class = np.array(y_pred_class).round(2).astype(str)
    
    # Calculate Metrics
    acc = accuracy_score(y_true_class, y_pred_class)
    macro_f1 = f1_score(y_true_class, y_pred_class, average='macro')
    
    print("-" * 30)
    print(f"ZENN Classification Performance")
    print("-" * 30)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("-" * 30)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_class, y_pred_class))

if __name__ == "__main__":
    benchmark()

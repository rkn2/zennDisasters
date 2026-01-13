
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def benchmark_multiscale():
    # --- Full Scale (0-5) ---
    print("\n" + "="*40)
    print("FULL SCALE (0-5) RESULTS")
    print("="*40)
    
    y_true_full = pd.read_csv("y_full_labels.csv", header=None).values.flatten()
    y_pred_cont = np.loadtxt("full_zenn_predictions.txt")
    
    # Map predictions to 0-5
    # Input was norm / 5.0. Output is approx 0-1 range.
    # Prediction class = closest integer to (pred * 5) ? Or just use Nearest Neighbor to valid labels
    # Let's map pred back to 0-5 scale
    y_pred_scaled = y_pred_cont * 5.0
    y_pred_full = np.round(y_pred_scaled).astype(int)
    
    # Clip to valid range
    y_pred_full = np.clip(y_pred_full, 0, 5)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_full, y_pred_full))
    print("\nClassification Report:")
    print(classification_report(y_true_full, y_pred_full, zero_division=0))
    
    # --- Consolidated Scale (0-2) ---
    print("\n" + "="*40)
    print("CONSOLIDATED SCALE (0-2) RESULTS")
    print("="*40)
    
    y_true_cons = pd.read_csv("y_consol_labels.csv", header=None).values.flatten()
    y_pred_cont_cons = np.loadtxt("consol_zenn_predictions.txt")
    
    # Map predictions to 0-2
    # Input was norm / 2.0
    y_pred_scaled_cons = y_pred_cont_cons * 2.0
    y_pred_cons = np.round(y_pred_scaled_cons).astype(int)
    
    y_pred_cons = np.clip(y_pred_cons, 0, 2)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_cons, y_pred_cons))
    print("\nClassification Report:")
    print(classification_report(y_true_cons, y_pred_cons, zero_division=0))

if __name__ == "__main__":
    benchmark_multiscale()

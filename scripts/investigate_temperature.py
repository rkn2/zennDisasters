"""
Investigate Temperature Parameter in Hazard-Neutral Setting

This script compares two approaches for ZENN when tornado_EF is excluded:
1. No explicit T: Simply remove the T column entirely
2. T=1 (constant): Replace T column with constant value of 1.0

This is important because ZENN's physics-based framework uses "Temperature"
to drive phase transitions. When hazard is excluded, we need to decide
how to handle this conceptually.
"""

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import sys
sys.path.append('../scripts')
from zenn_real_model import CoupledModel, KLDivergenceLoss, device

# Config
SEED = 42
KFOLD = 5
SCALE_FACTOR = 2.0
np.random.seed(SEED)
torch.manual_seed(SEED)

def run_cv(X, y_norm, y_cls, approach_name):
    """Run 5-Fold CV and return metrics."""
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    
    accuracies, f1_scores = [], []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_norm = y_norm[train_idx]
        y_test_cls = y_cls[test_idx]
        
        # Train ZENN
        X_train_t = torch.tensor(X_train, dtype=torch.double).to(device)
        y_train_t = torch.tensor(y_train_norm[:, None], dtype=torch.double).to(device)
        X_test_t = torch.tensor(X_test, dtype=torch.double).to(device)
        
        model = CoupledModel(width=32, input_dim=X.shape[1], num_networks=8).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = KLDivergenceLoss()
        
        for _ in range(1500):
            optimizer.zero_grad()
            y_pred = model(X_train_t)
            loss = criterion(y_pred, y_train_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test_t).cpu().numpy().flatten()
        
        y_pred_cls = np.clip(np.round(y_pred_test * SCALE_FACTOR).astype(int), 0, int(SCALE_FACTOR))
        
        accuracies.append(accuracy_score(y_test_cls, y_pred_cls))
        f1_scores.append(f1_score(y_test_cls, y_pred_cls, average='macro', zero_division=0))
    
    return {
        'approach': approach_name,
        'acc_mean': np.mean(accuracies),
        'acc_std': np.std(accuracies),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'n_features': X.shape[1]
    }

def main():
    # Load Data
    X_full = pd.read_csv("../data/X_multiscale.csv", header=None).values
    y_norm = pd.read_csv("../data/y_consol.csv", header=None).values.flatten()
    y_cls = np.round(y_norm * SCALE_FACTOR).astype(int)
    
    # T is last column
    T_idx = X_full.shape[1] - 1
    n_samples = X_full.shape[0]
    
    print("="*70)
    print("INVESTIGATING TEMPERATURE PARAMETER IN HAZARD-NEUTRAL SETTING")
    print("="*70)
    print(f"Dataset: {n_samples} samples, {X_full.shape[1]} features")
    print(f"T column index: {T_idx}")
    print()
    
    results = []
    
    # --- Approach 1: Hazard-Inclusive (Baseline) ---
    print("Running: Hazard-Inclusive (T = tornado_EF) [Baseline]")
    result1 = run_cv(X_full, y_norm, y_cls, "Hazard-Inclusive (T=tornado_EF)")
    results.append(result1)
    print(f"  Accuracy: {result1['acc_mean']:.4f} ± {result1['acc_std']:.4f}")
    print(f"  Macro F1: {result1['f1_mean']:.4f} ± {result1['f1_std']:.4f}")
    print()
    
    # --- Approach 2: No Explicit T (remove column) ---
    print("Running: No Explicit T (remove T column entirely)")
    X_no_T = np.delete(X_full, T_idx, axis=1)
    result2 = run_cv(X_no_T, y_norm, y_cls, "No Explicit T (removed)")
    results.append(result2)
    print(f"  Accuracy: {result2['acc_mean']:.4f} ± {result2['acc_std']:.4f}")
    print(f"  Macro F1: {result2['f1_mean']:.4f} ± {result2['f1_std']:.4f}")
    print()
    
    # --- Approach 3: T = 1.0 (constant high temperature) ---
    print("Running: T = 1.0 (constant high temperature)")
    X_T1 = X_full.copy()
    X_T1[:, T_idx] = 1.0
    result3 = run_cv(X_T1, y_norm, y_cls, "Constant T=1.0")
    results.append(result3)
    print(f"  Accuracy: {result3['acc_mean']:.4f} ± {result3['acc_std']:.4f}")
    print(f"  Macro F1: {result3['f1_mean']:.4f} ± {result3['f1_std']:.4f}")
    print()
    
    # --- Approach 4: T = 0.5 (constant medium temperature) ---
    print("Running: T = 0.5 (constant medium temperature)")
    X_T05 = X_full.copy()
    X_T05[:, T_idx] = 0.5
    result4 = run_cv(X_T05, y_norm, y_cls, "Constant T=0.5")
    results.append(result4)
    print(f"  Accuracy: {result4['acc_mean']:.4f} ± {result4['acc_std']:.4f}")
    print(f"  Macro F1: {result4['f1_mean']:.4f} ± {result4['f1_std']:.4f}")
    print()
    
    # --- Summary ---
    print("="*70)
    print("SUMMARY: Temperature Parameter Comparison")
    print("="*70)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv("../outputs/temperature_comparison.csv", index=False)
    print("\nSaved to outputs/temperature_comparison.csv")

if __name__ == "__main__":
    main()


import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from zenn_real_model import CoupledModel, KLDivergenceLoss, device
import sys

# Seed for reproducibility in splits
SEED = 42

def run_cv(target_file="real_tornado_targets.csv", k=5):
    print(f"\n{'='*40}")
    print(f"Running {k}-Fold Cross-Validation")
    print(f"Target: {target_file}")
    print(f"{'='*40}\n")
    
    # Load Data
    X_all = pd.read_csv("X_multiscale.csv", header=None).values
    y_all = pd.read_csv(target_file, header=None).values.flatten()
    
    # Check if target is norm or raw
    # If using y_full.csv or y_consol.csv, they are normalized. 
    # To measure classification metrics, we must map back to classes OR map predictions to norm values.
    # Strategy: Train on NORM, Predict NORM, then Map Both to Classes for Metrics.
    
    # Identify Scale
    if "full" in target_file:
        scale_factor = 5.0
        clip_max = 5
    elif "consol" in target_file:
        scale_factor = 2.0
        clip_max = 2
    else:
        # Default or unknown - guess based on max value?
        # If max <= 1.0, assume it is normalized.
        # But we need to know the factor. 
        # Using 5.0 as default for 'real_tornado_targets.csv'
        scale_factor = 5.0
        clip_max = 5
        
    print(f"Assumed Scale Factor: {scale_factor} (Max Class: {clip_max})")

    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        print(f"\n--- Fold {fold+1}/{k} ---")
        
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        
        # Tensors
        X_train_t = torch.tensor(X_train, dtype=torch.double).to(device)
        y_train_t = torch.tensor(y_train[:, None], dtype=torch.double).to(device)
        X_test_t = torch.tensor(X_test, dtype=torch.double).to(device)
        # y_test is kept numpy for sklearn metrics evaluation
        
        # Init Model (Fresh each fold)
        input_dim = X_train.shape[1]
        model = CoupledModel(width=32, input_dim=input_dim, num_networks=8).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = KLDivergenceLoss()
        
        # Training Loop
        epochs = 2000 # Enough for convergence based on previous runs
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = model(X_train_t)
            loss = criterion(y_pred, y_train_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred_test_norm = model(X_test_t).cpu().numpy().flatten()
            
        # Convert to Classes
        y_test_class = np.round(y_test * scale_factor).astype(int)
        y_pred_class = np.round(y_pred_test_norm * scale_factor).astype(int)
        
        y_pred_class = np.clip(y_pred_class, 0, clip_max)
        
        # Metrics
        acc = accuracy_score(y_test_class, y_pred_class)
        f1 = f1_score(y_test_class, y_pred_class, average='macro', zero_division=0)
        
        print(f"Fold {fold+1} Acc: {acc:.4f} | Macro F1: {f1:.4f}")
        fold_metrics.append({'acc': acc, 'f1': f1})
        
    # Average Metrics
    avg_acc = np.mean([m['acc'] for m in fold_metrics])
    avg_f1 = np.mean([m['f1'] for m in fold_metrics])
    
    print("\n" + "="*40)
    print(f"CV RESULTS ({k}-Fold) - {target_file}")
    print("="*40)
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average Macro F1: {avg_f1:.4f}")
    print("="*40)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cv(sys.argv[1])
    else:
        run_cv() 


import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from zenn_real_model import CoupledModel, KLDivergenceLoss, device
import matplotlib.pyplot as plt
import seaborn as sns

# Config
SEED = 42
KFOLD = 5
TARGET_FILE = "y_consol.csv"
SCALE_FACTOR = 2.0
CLIP_MAX = 2

def run_comparison():
    print(f"Loading Data: {TARGET_FILE}")
    X_all = pd.read_csv("X_multiscale.csv", header=None).values
    y_raw = pd.read_csv(TARGET_FILE, header=None).values.flatten()
    
    # Map normalized y back to classes 0,1,2
    y_classes = np.round(y_raw * SCALE_FACTOR).astype(int)
    
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    
    # Accumulators
    zenn_true_all = []
    zenn_pred_all = []
    
    xgb_true_all = []
    xgb_pred_all = []
    
    print(f"Running {KFOLD}-Fold CV for ZENN and XGBoost...")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        print(f"Fold {fold+1}/{KFOLD}")
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train_norm, y_test_norm = y_raw[train_idx], y_raw[test_idx] # For ZENN
        y_train_cls, y_test_cls = y_classes[train_idx], y_classes[test_idx] # For XGBoost
        
        # --- ZENN ---
        X_train_t = torch.tensor(X_train, dtype=torch.double).to(device)
        y_train_t = torch.tensor(y_train_norm[:, None], dtype=torch.double).to(device)
        X_test_t = torch.tensor(X_test, dtype=torch.double).to(device)
        
        input_dim = X_train.shape[1]
        model = CoupledModel(width=32, input_dim=input_dim, num_networks=8).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = KLDivergenceLoss()
        
        # Train ZENN
        epochs = 1500 # Slightly reduced for speed, converged ~1000 usually
        for _ in range(epochs):
            optimizer.zero_grad()
            y_p = model(X_train_t)
            loss = criterion(y_p, y_train_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            y_p_test_norm = model(X_test_t).cpu().numpy().flatten()
            
        y_p_test_cls = np.round(y_p_test_norm * SCALE_FACTOR).astype(int)
        y_p_test_cls = np.clip(y_p_test_cls, 0, CLIP_MAX)
        
        zenn_true_all.extend(y_test_cls)
        zenn_pred_all.extend(y_p_test_cls)
        
        # --- XGBoost ---
        # "Hazard-Inclusive" approx params: n_estimators=100, max_depth=6 (Default is fine for baseline)
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=SEED)
        xgb.fit(X_train, y_train_cls)
        y_xgb_pred = xgb.predict(X_test)
        
        xgb_true_all.extend(y_test_cls)
        xgb_pred_all.extend(y_xgb_pred)

    # Compute Confusion Matrices
    cm_zenn = confusion_matrix(zenn_true_all, zenn_pred_all)
    cm_xgb = confusion_matrix(xgb_true_all, xgb_pred_all)
    
    acc_zenn = accuracy_score(zenn_true_all, zenn_pred_all)
    acc_xgb = accuracy_score(xgb_true_all, xgb_pred_all)
    
    print("\n" + "="*40)
    print(f"ZENN CV Confusion Matrix (Acc: {acc_zenn:.4f})")
    print(cm_zenn)
    print("\n" + "="*40)
    print(f"XGBoost CV Confusion Matrix (Acc: {acc_xgb:.4f})")
    print(cm_xgb)
    print("="*40)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title(f'XGBoost Confusion Matrix\nAccuracy: {acc_xgb:.2f}')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_xticklabels(['Undamaged', 'Minor/Mod', 'Severe'])
    axes[0].set_yticklabels(['Undamaged', 'Minor/Mod', 'Severe'])

    sns.heatmap(cm_zenn, annot=True, fmt='d', cmap='Oranges', ax=axes[1], cbar=False)
    axes[1].set_title(f'ZENN Confusion Matrix\nAccuracy: {acc_zenn:.2f}')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_xticklabels(['Undamaged', 'Minor/Mod', 'Severe'])
    axes[1].set_yticklabels(['Undamaged', 'Minor/Mod', 'Severe'])
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_comparison.png')
    print("Saved plot to confusion_matrix_comparison.png")

if __name__ == "__main__":
    run_comparison()

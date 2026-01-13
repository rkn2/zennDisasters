
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from xgboost import XGBClassifier
from zenn_real_model import CoupledModel, KLDivergenceLoss, device
import matplotlib.pyplot as plt
import seaborn as sns

# Config
SEED = 42
KFOLD = 5
SCALE_FACTOR = 2.0
CLIP_MAX = 2

def run_cv_experiment(X, y_norm, y_cls, setting_name):
    """Run 5-Fold CV for both ZENN and XGBoost, return confusion matrices."""
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    
    zenn_true_all, zenn_pred_all = [], []
    xgb_true_all, xgb_pred_all = [], []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"  Fold {fold+1}/{KFOLD}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_norm, y_test_norm = y_norm[train_idx], y_norm[test_idx]
        y_train_cls, y_test_cls = y_cls[train_idx], y_cls[test_idx]
        
        # --- ZENN ---
        X_train_t = torch.tensor(X_train, dtype=torch.double).to(device)
        y_train_t = torch.tensor(y_train_norm[:, None], dtype=torch.double).to(device)
        X_test_t = torch.tensor(X_test, dtype=torch.double).to(device)
        
        model = CoupledModel(width=32, input_dim=X_train.shape[1], num_networks=8).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = KLDivergenceLoss()
        
        for _ in range(1500):
            optimizer.zero_grad()
            y_p = model(X_train_t)
            loss = criterion(y_p, y_train_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            y_p_test = model(X_test_t).cpu().numpy().flatten()
            
        y_p_cls = np.clip(np.round(y_p_test * SCALE_FACTOR).astype(int), 0, CLIP_MAX)
        zenn_true_all.extend(y_test_cls)
        zenn_pred_all.extend(y_p_cls)
        
        # --- XGBoost ---
        xgb = XGBClassifier(eval_metric='mlogloss', random_state=SEED, verbosity=0)
        xgb.fit(X_train, y_train_cls)
        y_xgb_pred = xgb.predict(X_test)
        
        xgb_true_all.extend(y_test_cls)
        xgb_pred_all.extend(y_xgb_pred)
    
    # Confusion Matrices
    cm_zenn = confusion_matrix(zenn_true_all, zenn_pred_all)
    cm_xgb = confusion_matrix(xgb_true_all, xgb_pred_all)
    
    acc_zenn = accuracy_score(zenn_true_all, zenn_pred_all)
    acc_xgb = accuracy_score(xgb_true_all, xgb_pred_all)
    f1_zenn = f1_score(zenn_true_all, zenn_pred_all, average='macro', zero_division=0)
    f1_xgb = f1_score(xgb_true_all, xgb_pred_all, average='macro', zero_division=0)
    
    return {
        'cm_zenn': cm_zenn, 'cm_xgb': cm_xgb,
        'acc_zenn': acc_zenn, 'acc_xgb': acc_xgb,
        'f1_zenn': f1_zenn, 'f1_xgb': f1_xgb
    }

def main():
    # Load Data
    X_full = pd.read_csv("X_multiscale.csv", header=None).values
    y_norm = pd.read_csv("y_consol.csv", header=None).values.flatten()
    y_cls = np.round(y_norm * SCALE_FACTOR).astype(int)
    
    # T is last column
    T_idx = X_full.shape[1] - 1
    X_neutral = np.delete(X_full, T_idx, axis=1)
    
    # --- HAZARD-INCLUSIVE ---
    print("="*50)
    print("HAZARD-INCLUSIVE (With Tornado EF)")
    print("="*50)
    results_hi = run_cv_experiment(X_full, y_norm, y_cls, "Hazard-Inclusive")
    
    # --- HAZARD-NEUTRAL ---
    print("\n" + "="*50)
    print("HAZARD-NEUTRAL (Without Tornado EF)")
    print("="*50)
    results_hn = run_cv_experiment(X_neutral, y_norm, y_cls, "Hazard-Neutral")
    
    # --- PLOT ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    labels = ['Undamaged', 'Minor/Mod', 'Severe']
    
    # Row 1: Hazard-Inclusive
    sns.heatmap(results_hi['cm_xgb'], annot=True, fmt='d', cmap='Blues', ax=axes[0,0], cbar=False)
    axes[0,0].set_title(f"XGBoost (Hazard-Inclusive)\nAcc: {results_hi['acc_xgb']:.2f}, F1: {results_hi['f1_xgb']:.2f}")
    axes[0,0].set_xticklabels(labels); axes[0,0].set_yticklabels(labels)
    axes[0,0].set_xlabel('Predicted'); axes[0,0].set_ylabel('True')
    
    sns.heatmap(results_hi['cm_zenn'], annot=True, fmt='d', cmap='Oranges', ax=axes[0,1], cbar=False)
    axes[0,1].set_title(f"ZENN (Hazard-Inclusive)\nAcc: {results_hi['acc_zenn']:.2f}, F1: {results_hi['f1_zenn']:.2f}")
    axes[0,1].set_xticklabels(labels); axes[0,1].set_yticklabels(labels)
    axes[0,1].set_xlabel('Predicted'); axes[0,1].set_ylabel('True')
    
    # Row 2: Hazard-Neutral
    sns.heatmap(results_hn['cm_xgb'], annot=True, fmt='d', cmap='Blues', ax=axes[1,0], cbar=False)
    axes[1,0].set_title(f"XGBoost (Hazard-Neutral)\nAcc: {results_hn['acc_xgb']:.2f}, F1: {results_hn['f1_xgb']:.2f}")
    axes[1,0].set_xticklabels(labels); axes[1,0].set_yticklabels(labels)
    axes[1,0].set_xlabel('Predicted'); axes[1,0].set_ylabel('True')
    
    sns.heatmap(results_hn['cm_zenn'], annot=True, fmt='d', cmap='Oranges', ax=axes[1,1], cbar=False)
    axes[1,1].set_title(f"ZENN (Hazard-Neutral)\nAcc: {results_hn['acc_zenn']:.2f}, F1: {results_hn['f1_zenn']:.2f}")
    axes[1,1].set_xticklabels(labels); axes[1,1].set_yticklabels(labels)
    axes[1,1].set_xlabel('Predicted'); axes[1,1].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_hazard_comparison.png', dpi=150)
    print("\nSaved to confusion_matrix_hazard_comparison.png")
    
    # --- SUMMARY TABLE ---
    print("\n" + "="*60)
    print("SUMMARY: Accuracy and Macro F1 by Setting")
    print("="*60)
    print(f"{'Setting':<20} {'Model':<10} {'Accuracy':<12} {'Macro F1':<12}")
    print("-"*54)
    print(f"{'Hazard-Inclusive':<20} {'XGBoost':<10} {results_hi['acc_xgb']:<12.4f} {results_hi['f1_xgb']:<12.4f}")
    print(f"{'Hazard-Inclusive':<20} {'ZENN':<10} {results_hi['acc_zenn']:<12.4f} {results_hi['f1_zenn']:<12.4f}")
    print(f"{'Hazard-Neutral':<20} {'XGBoost':<10} {results_hn['acc_xgb']:<12.4f} {results_hn['f1_xgb']:<12.4f}")
    print(f"{'Hazard-Neutral':<20} {'ZENN':<10} {results_hn['acc_zenn']:<12.4f} {results_hn['f1_zenn']:<12.4f}")

if __name__ == "__main__":
    main()

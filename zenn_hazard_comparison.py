
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from zenn_real_model import CoupledModel, KLDivergenceLoss, device
import matplotlib.pyplot as plt

# Config
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def train_model(X, y, epochs=1500):
    """Train a ZENN model and return it."""
    X_t = torch.tensor(X, dtype=torch.double).to(device)
    y_t = torch.tensor(y[:, None], dtype=torch.double).to(device)
    
    model = CoupledModel(width=32, input_dim=X.shape[1], num_networks=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = KLDivergenceLoss()
    
    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_t)
        loss = criterion(y_pred, y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    return model

def permutation_importance(model, X, y, scale_factor=2.0, n_repeats=5):
    """Calculate permutation importance for each feature."""
    X_t = torch.tensor(X, dtype=torch.double).to(device)
    
    model.eval()
    with torch.no_grad():
        y_pred_base = model(X_t).cpu().numpy().flatten()
    
    y_pred_class = np.clip(np.round(y_pred_base * scale_factor).astype(int), 0, int(scale_factor))
    y_true_class = np.round(y * scale_factor).astype(int)
    base_acc = accuracy_score(y_true_class, y_pred_class)
    
    importances = []
    n_features = X.shape[1]
    
    for feat_idx in range(n_features):
        acc_drops = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feat_idx])
            
            X_perm_t = torch.tensor(X_permuted, dtype=torch.double).to(device)
            with torch.no_grad():
                y_pred_perm = model(X_perm_t).cpu().numpy().flatten()
            
            y_pred_perm_class = np.clip(np.round(y_pred_perm * scale_factor).astype(int), 0, int(scale_factor))
            perm_acc = accuracy_score(y_true_class, y_pred_perm_class)
            acc_drops.append(base_acc - perm_acc)
        
        importances.append(np.mean(acc_drops))
    
    return np.array(importances)

def main():
    # Load Data
    X_full = pd.read_csv("X_multiscale.csv", header=None).values
    y = pd.read_csv("y_consol.csv", header=None).values.flatten()
    
    # Load feature names
    feature_names = open("real_feature_names.txt").read().strip().split("\n")
    if len(feature_names) < X_full.shape[1]:
        feature_names.append("T")
    
    # Identify T column (last column)
    T_idx = X_full.shape[1] - 1
    
    # --- HAZARD-INCLUSIVE (with T) ---
    print("="*50)
    print("ZENN HAZARD-INCLUSIVE (With Tornado EF)")
    print("="*50)
    
    model_hi = train_model(X_full, y)
    perm_hi = permutation_importance(model_hi, X_full, y)
    
    results_hi = pd.DataFrame({
        'Feature': feature_names[:X_full.shape[1]],
        'Importance': perm_hi
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Features (Hazard-Inclusive):")
    print(results_hi.head(10).to_string(index=False))
    results_hi.to_csv("zenn_importance_hazard_inclusive.csv", index=False)
    
    # --- HAZARD-NEUTRAL (without T) ---
    print("\n" + "="*50)
    print("ZENN HAZARD-NEUTRAL (Without Tornado EF)")
    print("="*50)
    
    # Remove T column
    X_neutral = np.delete(X_full, T_idx, axis=1)
    feature_names_neutral = feature_names[:T_idx]  # Exclude T
    
    model_hn = train_model(X_neutral, y)
    perm_hn = permutation_importance(model_hn, X_neutral, y)
    
    results_hn = pd.DataFrame({
        'Feature': feature_names_neutral,
        'Importance': perm_hn
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Features (Hazard-Neutral):")
    print(results_hn.head(10).to_string(index=False))
    results_hn.to_csv("zenn_importance_hazard_neutral.csv", index=False)
    
    # --- COMPARISON TABLE ---
    print("\n" + "="*50)
    print("COMPARISON: XGBoost vs ZENN")
    print("="*50)
    
    # Load XGBoost results
    xgb_hi = pd.read_csv("xgb_hi_importance.csv")
    xgb_hn = pd.read_csv("xgb_hn_importance.csv")
    
    print("\n--- HAZARD-INCLUSIVE ---")
    print(f"{'Rank':<5} {'XGBoost':<30} {'ZENN':<30}")
    print("-"*65)
    for i in range(min(10, len(results_hi))):
        xgb_feat = xgb_hi.iloc[i]['Feature'] if i < len(xgb_hi) else '-'
        zenn_feat = results_hi.iloc[i]['Feature']
        print(f"{i+1:<5} {xgb_feat:<30} {zenn_feat:<30}")
    
    print("\n--- HAZARD-NEUTRAL ---")
    print(f"{'Rank':<5} {'XGBoost':<30} {'ZENN':<30}")
    print("-"*65)
    for i in range(min(10, len(results_hn))):
        xgb_feat = xgb_hn.iloc[i]['Feature'] if i < len(xgb_hn) else '-'
        zenn_feat = results_hn.iloc[i]['Feature']
        print(f"{i+1:<5} {xgb_feat:<30} {zenn_feat:<30}")

if __name__ == "__main__":
    main()

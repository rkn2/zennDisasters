
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

def train_model(X, y, epochs=2000):
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

def permutation_importance(model, X, y, scale_factor=2.0, n_repeats=10):
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

def gradient_sensitivity(model, X):
    """Calculate gradient-based feature sensitivity."""
    X_t = torch.tensor(X, dtype=torch.double, requires_grad=True).to(device)
    
    model.eval()
    y_pred = model(X_t)
    
    # Backprop to get gradients w.r.t. inputs
    y_pred.sum().backward()
    
    # Mean absolute gradient per feature
    grads = X_t.grad.cpu().numpy()
    mean_abs_grads = np.mean(np.abs(grads), axis=0)
    
    return mean_abs_grads

def main():
    # Load Data
    X = pd.read_csv("X_multiscale.csv", header=None).values
    y = pd.read_csv("y_consol.csv", header=None).values.flatten()
    
    # Load feature names
    feature_names = open("real_feature_names.txt").read().strip().split("\n")
    # Append 'T' as it was added last
    if len(feature_names) < X.shape[1]:
        feature_names.append("T (tornado_EF)")
    
    print(f"Training ZENN model on full dataset...")
    model = train_model(X, y, epochs=1500)
    
    print("Calculating Permutation Importance...")
    perm_imp = permutation_importance(model, X, y, scale_factor=2.0, n_repeats=5)
    
    print("Calculating Gradient Sensitivity...")
    grad_sens = gradient_sensitivity(model, X)
    
    # Create DataFrame
    results = pd.DataFrame({
        'Feature': feature_names[:X.shape[1]],
        'Permutation_Importance': perm_imp,
        'Gradient_Sensitivity': grad_sens
    })
    
    # Sort by Permutation Importance
    results = results.sort_values('Permutation_Importance', ascending=False)
    
    print("\n" + "="*60)
    print("ZENN FEATURE IMPORTANCE RESULTS")
    print("="*60)
    print(results.to_string(index=False))
    
    # Save
    results.to_csv("zenn_feature_importance.csv", index=False)
    print("\nSaved to zenn_feature_importance.csv")
    
    # Plot Top 15
    top_n = 15
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Permutation Importance
    top_perm = results.head(top_n)
    axes[0].barh(top_perm['Feature'], top_perm['Permutation_Importance'], color='steelblue')
    axes[0].set_xlabel('Accuracy Drop')
    axes[0].set_title('ZENN Permutation Importance (Top 15)')
    axes[0].invert_yaxis()
    
    # Gradient Sensitivity
    results_grad = results.sort_values('Gradient_Sensitivity', ascending=False)
    top_grad = results_grad.head(top_n)
    axes[1].barh(top_grad['Feature'], top_grad['Gradient_Sensitivity'], color='darkorange')
    axes[1].set_xlabel('Mean |Gradient|')
    axes[1].set_title('ZENN Gradient Sensitivity (Top 15)')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('zenn_feature_importance.png')
    print("Saved plot to zenn_feature_importance.png")

if __name__ == "__main__":
    main()


import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from scipy.stats import wilcoxon, ttest_rel
from zenn_real_model import CoupledModel, KLDivergenceLoss, device

# Config
SEED = 42
KFOLD = 5
SCALE_FACTOR = 2.0
CLIP_MAX = 2

def run_cv_folds(X, y_norm, y_cls, setting_name):
    """Run 5-Fold CV and return per-fold metrics."""
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    
    zenn_acc, zenn_f1 = [], []
    xgb_acc, xgb_f1 = [], []
    
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
        
        zenn_acc.append(accuracy_score(y_test_cls, y_p_cls))
        zenn_f1.append(f1_score(y_test_cls, y_p_cls, average='macro', zero_division=0))
        
        # --- XGBoost ---
        xgb = XGBClassifier(eval_metric='mlogloss', random_state=SEED, verbosity=0)
        xgb.fit(X_train, y_train_cls)
        y_xgb_pred = xgb.predict(X_test)
        
        xgb_acc.append(accuracy_score(y_test_cls, y_xgb_pred))
        xgb_f1.append(f1_score(y_test_cls, y_xgb_pred, average='macro', zero_division=0))
    
    return {
        'zenn_acc': np.array(zenn_acc), 'zenn_f1': np.array(zenn_f1),
        'xgb_acc': np.array(xgb_acc), 'xgb_f1': np.array(xgb_f1)
    }

def statistical_test(a, b, metric_name):
    """Perform paired Wilcoxon test and report results."""
    diff = a - b
    mean_diff = np.mean(diff)
    
    # Wilcoxon signed-rank test (non-parametric paired test)
    try:
        stat, p_wilcoxon = wilcoxon(a, b)
    except ValueError:
        # If all differences are zero
        p_wilcoxon = 1.0
    
    # Paired t-test for reference
    stat_t, p_ttest = ttest_rel(a, b)
    
    return {
        'metric': metric_name,
        'mean_a': np.mean(a),
        'mean_b': np.mean(b),
        'mean_diff': mean_diff,
        'p_wilcoxon': p_wilcoxon,
        'p_ttest': p_ttest,
        'equivalent': p_wilcoxon > 0.05
    }

def main():
    # Load Data
    X_full = pd.read_csv("X_multiscale.csv", header=None).values
    y_norm = pd.read_csv("y_consol.csv", header=None).values.flatten()
    y_cls = np.round(y_norm * SCALE_FACTOR).astype(int)
    
    # T is last column
    T_idx = X_full.shape[1] - 1
    X_neutral = np.delete(X_full, T_idx, axis=1)
    
    results = []
    
    # --- HAZARD-INCLUSIVE ---
    print("="*50)
    print("HAZARD-INCLUSIVE (With Tornado EF)")
    print("="*50)
    folds_hi = run_cv_folds(X_full, y_norm, y_cls, "Hazard-Inclusive")
    
    acc_test_hi = statistical_test(folds_hi['xgb_acc'], folds_hi['zenn_acc'], 'Accuracy')
    f1_test_hi = statistical_test(folds_hi['xgb_f1'], folds_hi['zenn_f1'], 'Macro F1')
    
    print(f"\nHazard-Inclusive Results (XGBoost vs ZENN):")
    print(f"  Accuracy: XGBoost={acc_test_hi['mean_a']:.4f}, ZENN={acc_test_hi['mean_b']:.4f}, p={acc_test_hi['p_wilcoxon']:.4f}")
    print(f"  Macro F1: XGBoost={f1_test_hi['mean_a']:.4f}, ZENN={f1_test_hi['mean_b']:.4f}, p={f1_test_hi['p_wilcoxon']:.4f}")
    
    results.append({
        'Setting': 'Hazard-Inclusive', 'Metric': 'Accuracy',
        'XGBoost_Mean': acc_test_hi['mean_a'], 'ZENN_Mean': acc_test_hi['mean_b'],
        'Diff': acc_test_hi['mean_diff'], 'p_value': acc_test_hi['p_wilcoxon'],
        'Equivalent': acc_test_hi['equivalent']
    })
    results.append({
        'Setting': 'Hazard-Inclusive', 'Metric': 'Macro F1',
        'XGBoost_Mean': f1_test_hi['mean_a'], 'ZENN_Mean': f1_test_hi['mean_b'],
        'Diff': f1_test_hi['mean_diff'], 'p_value': f1_test_hi['p_wilcoxon'],
        'Equivalent': f1_test_hi['equivalent']
    })
    
    # --- HAZARD-NEUTRAL ---
    print("\n" + "="*50)
    print("HAZARD-NEUTRAL (Without Tornado EF)")
    print("="*50)
    folds_hn = run_cv_folds(X_neutral, y_norm, y_cls, "Hazard-Neutral")
    
    acc_test_hn = statistical_test(folds_hn['xgb_acc'], folds_hn['zenn_acc'], 'Accuracy')
    f1_test_hn = statistical_test(folds_hn['xgb_f1'], folds_hn['zenn_f1'], 'Macro F1')
    
    print(f"\nHazard-Neutral Results (XGBoost vs ZENN):")
    print(f"  Accuracy: XGBoost={acc_test_hn['mean_a']:.4f}, ZENN={acc_test_hn['mean_b']:.4f}, p={acc_test_hn['p_wilcoxon']:.4f}")
    print(f"  Macro F1: XGBoost={f1_test_hn['mean_a']:.4f}, ZENN={f1_test_hn['mean_b']:.4f}, p={f1_test_hn['p_wilcoxon']:.4f}")
    
    results.append({
        'Setting': 'Hazard-Neutral', 'Metric': 'Accuracy',
        'XGBoost_Mean': acc_test_hn['mean_a'], 'ZENN_Mean': acc_test_hn['mean_b'],
        'Diff': acc_test_hn['mean_diff'], 'p_value': acc_test_hn['p_wilcoxon'],
        'Equivalent': acc_test_hn['equivalent']
    })
    results.append({
        'Setting': 'Hazard-Neutral', 'Metric': 'Macro F1',
        'XGBoost_Mean': f1_test_hn['mean_a'], 'ZENN_Mean': f1_test_hn['mean_b'],
        'Diff': f1_test_hn['mean_diff'], 'p_value': f1_test_hn['p_wilcoxon'],
        'Equivalent': f1_test_hn['equivalent']
    })
    
    # --- SUMMARY ---
    print("\n" + "="*60)
    print("STATISTICAL EQUIVALENCE SUMMARY (Wilcoxon p > 0.05 = Equivalent)")
    print("="*60)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv("zenn_statistical_equivalence.csv", index=False)
    print("\nSaved to zenn_statistical_equivalence.csv")

if __name__ == "__main__":
    main()

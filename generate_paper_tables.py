"""
Recreate Paper Tables and Figures with ZENN Results
Tables: 3, 4, 5, 6, 7, 8, A.11
Figures: 7, 8
"""

import pandas as pd
import numpy as np

# Load ZENN results
zenn_equiv = pd.read_csv("zenn_statistical_equivalence.csv")
zenn_hi_imp = pd.read_csv("zenn_importance_hazard_inclusive.csv")
zenn_hn_imp = pd.read_csv("zenn_importance_hazard_neutral.csv")

# Load Original XGBoost results
xgb_hi_imp = pd.read_csv("xgb_hi_importance.csv")
xgb_hn_imp = pd.read_csv("xgb_hn_importance.csv")

# Load original performance data
orig_perf = pd.read_csv("/Users/rebeccanapolitano/antigravityProjects/featImp/tornado_vulnerability_outputs/model_performance_cv.csv")
orig_equiv = pd.read_csv("/Users/rebeccanapolitano/antigravityProjects/featImp/tornado_vulnerability_outputs/statistical_equivalence.csv")
orig_perm = pd.read_csv("/Users/rebeccanapolitano/antigravityProjects/featImp/tornado_vulnerability_outputs/permutation_importance.csv")

# ============================================================================
# TABLE 3: Model Performance Summary (Mean ± Std across CV folds)
# ============================================================================
print("="*80)
print("TABLE 3: Model Performance Summary (Including ZENN)")
print("="*80)

# Original models
models = ['DecisionTree', 'RandomForest', 'LogisticRegression', 'RidgeClassifier', 'LinearSVC', 'XGBoost']
settings = ['Hazard-Neutral', 'Hazard-Inclusive']

table3_data = []
for setting in settings:
    for model in models:
        subset = orig_perf[(orig_perf['Setting'] == setting) & (orig_perf['Model'] == model)]
        acc_mean = subset['Accuracy'].mean()
        acc_std = subset['Accuracy'].std()
        f1_mean = subset['MacroF1'].mean()
        f1_std = subset['MacroF1'].std()
        table3_data.append({
            'Setting': setting,
            'Model': model,
            'Accuracy': f"{acc_mean:.3f} ± {acc_std:.3f}",
            'Macro F1': f"{f1_mean:.3f} ± {f1_std:.3f}"
        })

# Add ZENN (from our CV results)
# ZENN HI: 0.8314 acc, 0.5458 f1 (from compare script)
# ZENN HN: 0.8023 acc, 0.4545 f1
table3_data.append({'Setting': 'Hazard-Inclusive', 'Model': 'ZENN', 'Accuracy': '0.831 ± 0.041', 'Macro F1': '0.515 ± 0.065'})
table3_data.append({'Setting': 'Hazard-Neutral', 'Model': 'ZENN', 'Accuracy': '0.802 ± 0.035', 'Macro F1': '0.438 ± 0.058'})

table3 = pd.DataFrame(table3_data)
print(table3.to_string(index=False))
table3.to_csv("paper_table3_with_zenn.csv", index=False)

# ============================================================================
# TABLE 4: Statistical Equivalence Testing (Including ZENN)
# ============================================================================
print("\n" + "="*80)
print("TABLE 4: Statistical Equivalence Testing (Including ZENN)")
print("="*80)

table4_data = []
# Original equivalence tests
for _, row in orig_equiv.iterrows():
    table4_data.append({
        'Setting': row['Setting'],
        'Model': row['Model'],
        'Best Model': row['Best_Model'],
        'p-value': f"{row['p_value']:.4f}",
        'Diff Mean': f"{row['Diff_Mean']:.4f}",
        'Equivalent': 'Yes' if row['p_value'] > 0.05 else 'No'
    })

# Add ZENN equivalence (vs XGBoost as reference)
for _, row in zenn_equiv.iterrows():
    if row['Metric'] == 'Accuracy':
        table4_data.append({
            'Setting': row['Setting'],
            'Model': 'ZENN',
            'Best Model': 'XGBoost',
            'p-value': f"{row['p_value']:.4f}",
            'Diff Mean': f"{row['Diff']:.4f}",
            'Equivalent': 'Yes' if row['Equivalent'] else 'No'
        })

table4 = pd.DataFrame(table4_data)
print(table4.to_string(index=False))
table4.to_csv("paper_table4_with_zenn.csv", index=False)

# ============================================================================
# TABLE 5 & 6: Top 10 Features Hazard-Inclusive (XGBoost, RF, ZENN)
# ============================================================================
print("\n" + "="*80)
print("TABLE 5: Top 10 Features - Hazard-Inclusive")
print("="*80)

# Get top 10 from each model for HI
rf_hi = orig_perm[(orig_perm['Setting']=='Hazard-Inclusive') & (orig_perm['Model']=='RandomForest')].groupby('Feature')['Decrease_in_Accuracy'].mean().sort_values(ascending=False).head(10)
xgb_hi = orig_perm[(orig_perm['Setting']=='Hazard-Inclusive') & (orig_perm['Model']=='XGBoost')].groupby('Feature')['Decrease_in_Accuracy'].mean().sort_values(ascending=False).head(10)
zenn_hi_top = zenn_hi_imp.head(10)

table5 = pd.DataFrame({
    'Rank': range(1, 11),
    'RandomForest': rf_hi.index.tolist(),
    'XGBoost': xgb_hi.index.tolist(),
    'ZENN': zenn_hi_top['Feature'].tolist()
})
print(table5.to_string(index=False))
table5.to_csv("paper_table5_hazard_inclusive_features.csv", index=False)

# ============================================================================
# TABLE 6: Top 10 Features Hazard-Neutral (XGBoost, RF, ZENN)
# ============================================================================
print("\n" + "="*80)
print("TABLE 6: Top 10 Features - Hazard-Neutral")
print("="*80)

rf_hn = orig_perm[(orig_perm['Setting']=='Hazard-Neutral') & (orig_perm['Model']=='RandomForest')].groupby('Feature')['Decrease_in_Accuracy'].mean().sort_values(ascending=False).head(10)
xgb_hn = orig_perm[(orig_perm['Setting']=='Hazard-Neutral') & (orig_perm['Model']=='XGBoost')].groupby('Feature')['Decrease_in_Accuracy'].mean().sort_values(ascending=False).head(10)
zenn_hn_top = zenn_hn_imp.head(10)

table6 = pd.DataFrame({
    'Rank': range(1, 11),
    'RandomForest': rf_hn.index.tolist(),
    'XGBoost': xgb_hn.index.tolist(),
    'ZENN': zenn_hn_top['Feature'].tolist()
})
print(table6.to_string(index=False))
table6.to_csv("paper_table6_hazard_neutral_features.csv", index=False)

# ============================================================================
# TABLE 7: Feature Overlap Analysis (Unique to each, Shared)
# ============================================================================
print("\n" + "="*80)
print("TABLE 7: Feature Overlap Analysis")
print("="*80)

# Hazard-Inclusive
rf_hi_set = set(rf_hi.index)
xgb_hi_set = set(xgb_hi.index)
zenn_hi_set = set(zenn_hi_top['Feature'])

shared_hi_all = rf_hi_set & xgb_hi_set & zenn_hi_set
unique_zenn_hi = zenn_hi_set - rf_hi_set - xgb_hi_set
unique_xgb_hi = xgb_hi_set - rf_hi_set - zenn_hi_set
unique_rf_hi = rf_hi_set - xgb_hi_set - zenn_hi_set

# Hazard-Neutral
rf_hn_set = set(rf_hn.index)
xgb_hn_set = set(xgb_hn.index)
zenn_hn_set = set(zenn_hn_top['Feature'])

shared_hn_all = rf_hn_set & xgb_hn_set & zenn_hn_set
unique_zenn_hn = zenn_hn_set - rf_hn_set - xgb_hn_set
unique_xgb_hn = xgb_hn_set - rf_hn_set - zenn_hn_set
unique_rf_hn = rf_hn_set - xgb_hn_set - zenn_hn_set

table7 = pd.DataFrame({
    'Setting': ['Hazard-Inclusive', 'Hazard-Inclusive', 'Hazard-Inclusive', 'Hazard-Inclusive',
                'Hazard-Neutral', 'Hazard-Neutral', 'Hazard-Neutral', 'Hazard-Neutral'],
    'Category': ['Shared (All 3)', 'Unique to RF', 'Unique to XGBoost', 'Unique to ZENN',
                 'Shared (All 3)', 'Unique to RF', 'Unique to XGBoost', 'Unique to ZENN'],
    'Count': [len(shared_hi_all), len(unique_rf_hi), len(unique_xgb_hi), len(unique_zenn_hi),
              len(shared_hn_all), len(unique_rf_hn), len(unique_xgb_hn), len(unique_zenn_hn)],
    'Features': [', '.join(shared_hi_all) if shared_hi_all else '-', 
                 ', '.join(unique_rf_hi) if unique_rf_hi else '-',
                 ', '.join(unique_xgb_hi) if unique_xgb_hi else '-',
                 ', '.join(unique_zenn_hi) if unique_zenn_hi else '-',
                 ', '.join(shared_hn_all) if shared_hn_all else '-',
                 ', '.join(unique_rf_hn) if unique_rf_hn else '-',
                 ', '.join(unique_xgb_hn) if unique_xgb_hn else '-',
                 ', '.join(unique_zenn_hn) if unique_zenn_hn else '-']
})
print(table7.to_string(index=False))
table7.to_csv("paper_table7_feature_overlap.csv", index=False)

# ============================================================================
# TABLE 8: ZENN vs XGBoost Direct Comparison
# ============================================================================
print("\n" + "="*80)
print("TABLE 8: ZENN vs XGBoost Direct Statistical Comparison")
print("="*80)

table8 = zenn_equiv.copy()
table8.to_csv("paper_table8_zenn_vs_xgboost.csv", index=False)
print(table8.to_string(index=False))

# ============================================================================
# TABLE A.11: Full Feature Importance Rankings (Appendix)
# ============================================================================
print("\n" + "="*80)
print("TABLE A.11: Full Feature Importance Rankings (All Models)")
print("="*80)

# Create full ranking table
all_features = set(zenn_hi_imp['Feature'].tolist() + zenn_hn_imp['Feature'].tolist())

full_rankings_hi = []
for feat in sorted(all_features):
    row = {'Feature': feat}
    
    # XGBoost HI
    xgb_rank = list(xgb_hi.index).index(feat) + 1 if feat in xgb_hi.index else '-'
    row['XGBoost_HI_Rank'] = xgb_rank
    
    # RF HI
    rf_rank = list(rf_hi.index).index(feat) + 1 if feat in rf_hi.index else '-'
    row['RF_HI_Rank'] = rf_rank
    
    # ZENN HI
    zenn_hi_ranks = zenn_hi_imp[zenn_hi_imp['Feature'] == feat]
    row['ZENN_HI_Rank'] = zenn_hi_ranks.index[0] + 1 if len(zenn_hi_ranks) > 0 else '-'
    row['ZENN_HI_Importance'] = zenn_hi_ranks['Importance'].values[0] if len(zenn_hi_ranks) > 0 else 0
    
    full_rankings_hi.append(row)

table_a11 = pd.DataFrame(full_rankings_hi).sort_values('ZENN_HI_Importance', ascending=False)
print(table_a11.head(20).to_string(index=False))
table_a11.to_csv("paper_table_a11_full_rankings.csv", index=False)

print("\n" + "="*80)
print("All tables saved to CSV files:")
print("  - paper_table3_with_zenn.csv")
print("  - paper_table4_with_zenn.csv")
print("  - paper_table5_hazard_inclusive_features.csv")
print("  - paper_table6_hazard_neutral_features.csv")
print("  - paper_table7_feature_overlap.csv")
print("  - paper_table8_zenn_vs_xgboost.csv")
print("  - paper_table_a11_full_rankings.csv")
print("="*80)

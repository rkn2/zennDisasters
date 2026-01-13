"""
Generate Figures 7 and 8 for Paper - Feature Importance Comparison Plots
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load data
zenn_hi = pd.read_csv("zenn_importance_hazard_inclusive.csv")
zenn_hn = pd.read_csv("zenn_importance_hazard_neutral.csv")
orig_perm = pd.read_csv("/Users/rebeccanapolitano/antigravityProjects/featImp/tornado_vulnerability_outputs/permutation_importance.csv")

# ============================================================================
# FIGURE 7: Top 10 Feature Importance Comparison - Hazard-Inclusive
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# RandomForest HI
rf_hi = orig_perm[(orig_perm['Setting']=='Hazard-Inclusive') & (orig_perm['Model']=='RandomForest')].groupby('Feature')['Decrease_in_Accuracy'].mean().sort_values(ascending=False).head(10)
axes[0].barh(range(10), rf_hi.values[::-1], color='forestgreen', alpha=0.8)
axes[0].set_yticks(range(10))
axes[0].set_yticklabels(rf_hi.index[::-1], fontsize=9)
axes[0].set_xlabel('Decrease in Accuracy')
axes[0].set_title('RandomForest\n(Hazard-Inclusive)', fontsize=12, fontweight='bold')
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# XGBoost HI
xgb_hi = orig_perm[(orig_perm['Setting']=='Hazard-Inclusive') & (orig_perm['Model']=='XGBoost')].groupby('Feature')['Decrease_in_Accuracy'].mean().sort_values(ascending=False).head(10)
axes[1].barh(range(10), xgb_hi.values[::-1], color='steelblue', alpha=0.8)
axes[1].set_yticks(range(10))
axes[1].set_yticklabels(xgb_hi.index[::-1], fontsize=9)
axes[1].set_xlabel('Decrease in Accuracy')
axes[1].set_title('XGBoost\n(Hazard-Inclusive)', fontsize=12, fontweight='bold')
axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# ZENN HI
zenn_hi_top = zenn_hi.head(10)
axes[2].barh(range(10), zenn_hi_top['Importance'].values[::-1], color='darkorange', alpha=0.8)
axes[2].set_yticks(range(10))
axes[2].set_yticklabels(zenn_hi_top['Feature'].values[::-1], fontsize=9)
axes[2].set_xlabel('Decrease in Accuracy')
axes[2].set_title('ZENN\n(Hazard-Inclusive)', fontsize=12, fontweight='bold')
axes[2].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.suptitle('Figure 7: Top 10 Feature Importance - Hazard-Inclusive Setting', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('paper_figure7_hazard_inclusive.png', dpi=150, bbox_inches='tight')
print("Saved: paper_figure7_hazard_inclusive.png")

# ============================================================================
# FIGURE 8: Top 10 Feature Importance Comparison - Hazard-Neutral
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# RandomForest HN
rf_hn = orig_perm[(orig_perm['Setting']=='Hazard-Neutral') & (orig_perm['Model']=='RandomForest')].groupby('Feature')['Decrease_in_Accuracy'].mean().sort_values(ascending=False).head(10)
axes[0].barh(range(10), rf_hn.values[::-1], color='forestgreen', alpha=0.8)
axes[0].set_yticks(range(10))
axes[0].set_yticklabels(rf_hn.index[::-1], fontsize=9)
axes[0].set_xlabel('Decrease in Accuracy')
axes[0].set_title('RandomForest\n(Hazard-Neutral)', fontsize=12, fontweight='bold')
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# XGBoost HN
xgb_hn = orig_perm[(orig_perm['Setting']=='Hazard-Neutral') & (orig_perm['Model']=='XGBoost')].groupby('Feature')['Decrease_in_Accuracy'].mean().sort_values(ascending=False).head(10)
axes[1].barh(range(10), xgb_hn.values[::-1], color='steelblue', alpha=0.8)
axes[1].set_yticks(range(10))
axes[1].set_yticklabels(xgb_hn.index[::-1], fontsize=9)
axes[1].set_xlabel('Decrease in Accuracy')
axes[1].set_title('XGBoost\n(Hazard-Neutral)', fontsize=12, fontweight='bold')
axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# ZENN HN
zenn_hn_top = zenn_hn.head(10)
axes[2].barh(range(10), zenn_hn_top['Importance'].values[::-1], color='darkorange', alpha=0.8)
axes[2].set_yticks(range(10))
axes[2].set_yticklabels(zenn_hn_top['Feature'].values[::-1], fontsize=9)
axes[2].set_xlabel('Decrease in Accuracy')
axes[2].set_title('ZENN\n(Hazard-Neutral)', fontsize=12, fontweight='bold')
axes[2].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.suptitle('Figure 8: Top 10 Feature Importance - Hazard-Neutral Setting', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('paper_figure8_hazard_neutral.png', dpi=150, bbox_inches='tight')
print("Saved: paper_figure8_hazard_neutral.png")

print("\nFigures 7 and 8 generated successfully!")

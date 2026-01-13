
import pandas as pd
import numpy as np

def preprocess_scales():
    # Load Data (Same logic as preprocess)
    csv_path = '/Users/rebeccanapolitano/antigravityProjects/featImp/updatedData/Revised_QuadState_Tornado_DataInput_pub - Copy_120525.csv'
    xlsx_path = '/Users/rebeccanapolitano/antigravityProjects/featImp/Nashville_Tornado_DataInput_Final_110725.xlsx'
    
    df_qs = pd.read_csv(csv_path)
    df_nash = pd.read_excel(xlsx_path)
    
    # Target
    cols = ['damage_indicator_u']
    
    df_qs = df_qs[cols]
    df_nash = df_nash[cols]
    
    df = pd.concat([df_qs, df_nash], axis=0, ignore_index=True)
    
    df['damage_indicator_u'] = pd.to_numeric(df['damage_indicator_u'], errors='coerce')
    df = df.dropna(subset=['damage_indicator_u'])
    
    # Filter valid range: 0-5
    # The anomalies 8, 17, 23 are likely data entry errors or archetype codes.
    # We will exclude them as we can't be sure of the mapping.
    df = df[df['damage_indicator_u'].isin([0, 1, 2, 3, 4, 5])]
    print(f"Valid samples (0-5 scale): {len(df)}")
    
    # --- Scale A: Full (0-5) ---
    # Normalize to 0-1 for ZENN
    y_full = df['damage_indicator_u'].values
    y_full_norm = y_full / 5.0 
    
    # --- Scale B: Consolidated (0-2) ---
    # User mentioned 0, 1, 2. Usually standard StEER ref:
    # 0 = Undamaged (0)
    # 1 = Minor/Moderate (1, 2)
    # 2 = Severe/Destruction (3, 4, 5)
    
    y_consol = df['damage_indicator_u'].replace({
        0: 0,
        1: 1, 2: 1,
        3: 2, 4: 2, 5: 2
    }).values
    
    y_consol_norm = y_consol / 2.0
    
    # Save targets aligned with features
    # Note: X must be filtered to match these valid rows!
    # I should re-run preprocess_real_data.py logic here or ensure indices match.
    # To avoid mismatch, I will perform the filtering ON the generated files from preprocess_real_data.py
    # But preprocess_real_data.py already saved X and y.
    # X and y saved there might include the anomalous rows.
    
    # STRATEGY: 
    # 1. Load the ALREADY GENERATED 'real_tornado_targets.csv' (y from preprocess_real_data)
    # 2. Load 'real_tornado_features.csv'
    # 3. Filter both based on y values.
    
    y_raw = pd.read_csv("real_tornado_targets.csv", header=None)
    X_raw = pd.read_csv("real_tornado_features.csv", header=None)
    
    # The 'damage' column in y_raw was normalized by max().
    # If max was 23 (anomaly), all values are skewed.
    # Let's check max value in y_raw to de-normalize?
    # Or just re-calculate from raw logic.
    
    # Better: Re-run the critical filtering logic on the raw data frames to ensure perfect alignment
    # Copied from preprocess_real_data.py mostly but with filtering:
    
    # (Re-loading raw again for safety)
    df_qs = pd.read_csv(csv_path)
    df_nash = pd.read_excel(xlsx_path)
    
    common_cols = list(set([
        'damage_indicator_u', 'tornado_EF', 'year_built_u', 'number_stories',
        'wall_thickness', 'roof_slope_u', 'parapet_height_m', 'wall_length_side',
        'wall_length_front', 'wall_fenesteration_per_front', 'roof_shape_u',
        'wall_substrate_u', 'retrofit_present_u'
    ]) & set(df_qs.columns) & set(df_nash.columns))
    
    df_qs = df_qs[common_cols]
    df_nash = df_nash[common_cols]
    df_merged = pd.concat([df_qs, df_nash], axis=0, ignore_index=True)
    
    # Cleaning Target
    df_merged = df_merged.dropna(subset=['damage_indicator_u'])
    df_merged['damage_indicator_u'] = pd.to_numeric(df_merged['damage_indicator_u'], errors='coerce')
    df_merged = df_merged.dropna(subset=['damage_indicator_u'])
    
    # FILTER ANOMALIES (0-5)
    df_merged = df_merged[df_merged['damage_indicator_u'].isin([0, 1, 2, 3, 4, 5])]
    print(f"Final valid dataset size: {len(df_merged)}")
    
    # --- Generate Targets ---
    # Full
    y_full = df_merged[['damage_indicator_u']].copy()
    y_full['norm'] = y_full['damage_indicator_u'] / 5.0
    
    # Consol
    y_consol = df_merged[['damage_indicator_u']].copy()
    y_consol['consol'] = y_consol['damage_indicator_u'].replace({
        0: 0,
        1: 1, 2: 1,
        3: 2, 4: 2, 5: 2
    })
    y_consol['norm'] = y_consol['consol'] / 2.0
    
    # --- Process Features (V+T) ---
    def parse_ef(val):
        if str(val).lower() == 'subef': return 0.0
        try:
             v = float(val)
             if v < 0: return 0.0
             return v
        except: return 0.0
        
    df_merged['tornado_EF'] = df_merged['tornado_EF'].apply(parse_ef)
    df_merged['T'] = (df_merged['tornado_EF'] + 1) / 6.0
    
    # Numeric Features
    num_cols = ['year_built_u', 'number_stories', 'wall_thickness', 
            'roof_slope_u', 'parapet_height_m', 'wall_length_side', 
            'wall_length_front', 'wall_fenesteration_per_front']
            
    for c in num_cols:
        if c in df_merged.columns:
            df_merged[c] = pd.to_numeric(df_merged[c], errors='coerce')
            df_merged[c] = df_merged[c].fillna(df_merged[c].median())
            if df_merged[c].max() > df_merged[c].min():
                df_merged[c] = (df_merged[c] - df_merged[c].min()) / (df_merged[c].max() - df_merged[c].min())
            else:
                df_merged[c] = 0.0
                
    # Categoricals
    cat_cols = ['roof_shape_u', 'wall_substrate_u', 'retrofit_present_u']
    existing_cat = [c for c in cat_cols if c in df_merged.columns]
    df_encoded = pd.get_dummies(df_merged, columns=existing_cat, dummy_na=True)
    
    # Select Features
    feature_cols = [c for c in df_encoded.columns if c not in ['damage_indicator_u', 'tornado_EF', 'T', 'consol', 'norm']]
    
    X = df_encoded[feature_cols].copy()
    X['T'] = df_encoded['T']
    
    # Force float
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
    X = X.astype(float)
    
    # SAVE
    X.to_csv('X_multiscale.csv', header=False, index=False)
    y_full['norm'].to_csv('y_full.csv', header=False, index=False)
    y_consol['norm'].to_csv('y_consol.csv', header=False, index=False)
    
    # Save raw class labels for benchmarking
    y_full['damage_indicator_u'].to_csv('y_full_labels.csv', header=False, index=False)
    y_consol['consol'].to_csv('y_consol_labels.csv', header=False, index=False)
    
    print("Saved X_multiscale.csv, y_full.csv, y_consol.csv, y_full_labels.csv, y_consol_labels.csv")

if __name__ == "__main__":
    preprocess_scales()


import pandas as pd
import numpy as np

def preprocess():
    # File Paths
    csv_path = '/Users/rebeccanapolitano/antigravityProjects/featImp/updatedData/Revised_QuadState_Tornado_DataInput_pub - Copy_120525.csv'
    xlsx_path = '/Users/rebeccanapolitano/antigravityProjects/featImp/Nashville_Tornado_DataInput_Final_110725.xlsx'
    
    # Load Data
    print("Loading data...")
    df_qs = pd.read_csv(csv_path)
    df_nash = pd.read_excel(xlsx_path)
    
    # Select Relevant Columns (Based on paper priority)
    # Target: damage_indicator_u (or degree_of_damage_u - check paper mapping)
    # T (Control): tornado_EF
    # V (Features): Structural & Geometric
    
    cols_to_keep = [
        # Target
        'damage_indicator_u', 
        
        # Hazard (T)
        'tornado_EF',
        
        # Features (V)
        'year_built_u',
        'number_stories',
        'wall_thickness',
        'roof_slope_u',
        'parapet_height_m',
        'wall_length_side',
        'wall_length_front',
        'wall_fenesteration_per_front', # Proxy for fenestration
        
        # Categoricals
        'roof_shape_u',
        'wall_substrate_u', # Paper mentions this
        'retrofit_present_u' # Paper mentions this
    ]
    
    # Normalize naming if different (Assuming mostly same based on previous inspection)
    # Only keep intersection for safety first
    common_cols = list(set(cols_to_keep) & set(df_qs.columns) & set(df_nash.columns))
    print(f"Common columns found: {len(common_cols)}/{len(cols_to_keep)}")
    print(f"Missing: {set(cols_to_keep) - set(common_cols)}")
    
    df_qs = df_qs[common_cols]
    df_nash = df_nash[common_cols]
    
    # Combine
    df = pd.concat([df_qs, df_nash], axis=0, ignore_index=True)
    print(f"Combined shape: {df.shape}")
    
    # --- Cleaning ---
    
    # Target: Map 'degree_of_damage_u' or 'damage_indicator_u' to 0-1
    # Assuming standard StEER: 0=None, 1=Minor, 2=Moderate, 3=Severe, 4=Destroyed
    # Normalize to [0, 1]
    
    # Only keep rows with valid target
    df = df.dropna(subset=['damage_indicator_u'])
    
    # Convert target to numeric (force coercion)
    df['damage_indicator_u'] = pd.to_numeric(df['damage_indicator_u'], errors='coerce')
    df = df.dropna(subset=['damage_indicator_u'])
    
    # Normalize Target
    max_dmg = df['damage_indicator_u'].max()
    df['damage'] = df['damage_indicator_u'] / max_dmg
    
    # Hazard (T): tornado_EF
    # Handle 'subEF' or non-numeric
    # Replace 'subEF' with 0 or -1? Paper says -1. ZENN needs continuous T.
    # Map subEF (-1) to 0.0, EF0 to 0.2, ..., EF5 to 1.0
    
    def parse_ef(val):
        if str(val).lower() == 'subef':
            return 0.0
        try:
            v = float(val)
            # Map -1 to 0 if present
            if v < 0: return 0.0
            return v
        except:
            return 0.0 # Default
            
    df['tornado_EF'] = df['tornado_EF'].apply(parse_ef)
    # Normalize EF (0-5) to [0.1, 1.0] to resemble temperature bounds
    # Avoid 0 for division stability
    df['T'] = (df['tornado_EF'] + 1) / 6.0 
    
    # Features (V)
    # 1. Clean Numerics
    num_cols = ['year_built_u', 'number_stories', 'wall_thickness', 
            'roof_slope_u', 'parapet_height_m', 'wall_length_side', 
            'wall_length_front', 'wall_fenesteration_per_front']
    
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            # Impute Median
            df[c] = df[c].fillna(df[c].median())
            # Normalize (Min-Max)
            if df[c].max() > df[c].min():
                df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
            else:
                df[c] = 0.0
    
    # 2. Clean Categoricals
    cat_cols = ['roof_shape_u', 'wall_substrate_u', 'retrofit_present_u']
    existing_cat = [c for c in cat_cols if c in df.columns]
    
    df = pd.get_dummies(df, columns=existing_cat, dummy_na=True)
    
    # --- Final Feature Assembly ---
    # X = [Features (V), Temperature (T)]
    
    # Identify V columns (All numeric columns except 'damage', 'damage_indicator_u', 'tornado_EF', 'T')
    exclude = ['damage', 'damage_indicator_u', 'tornado_EF', 'T']
    v_cols = [c for c in df.columns if c not in exclude]
    
    # Construct Output
    exclude = ['damage', 'damage_indicator_u', 'tornado_EF', 'T']
    v_cols = [c for c in df.columns if c not in exclude]
    
    X = df[v_cols].copy()
    X['T'] = df['T'] # Append T
    
    # Force Numeric (and cast bools to float)
    print("Converting to numeric and casting to float...")
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.astype(float) # Explicit cast for bools
        
    # Check for NaNs after coercion
    if X.isna().any().any():
        print("Warning: NaNs generated. Filling with 0.")
        X = X.fillna(0.0)
        
    # Verify dtypes
    print("X dtypes:")
    print(X.dtypes.value_counts())
    
    y = df[['damage']]

    
    # Save
    X.to_csv('real_tornado_features.csv', header=False, index=False)
    y.to_csv('real_tornado_targets.csv', header=False, index=False)
    
    # Save Feature Names for reference
    with open('real_feature_names.txt', 'w') as f:
        f.write('\n'.join(X.columns))

if __name__ == "__main__":
    preprocess()

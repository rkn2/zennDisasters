
import pandas as pd

def list_features():
    # Re-run the feature selection logic to get headers
    csv_path = '/Users/rebeccanapolitano/antigravityProjects/featImp/updatedData/Revised_QuadState_Tornado_DataInput_pub - Copy_120525.csv'
    xlsx_path = '/Users/rebeccanapolitano/antigravityProjects/featImp/Nashville_Tornado_DataInput_Final_110725.xlsx'
    
    df_qs = pd.read_csv(csv_path)
    df_nash = pd.read_excel(xlsx_path)
    
    # Common Cols Logic
    common_cols = list(set([
        'damage_indicator_u', 'tornado_EF', 'year_built_u', 'number_stories',
        'wall_thickness', 'roof_slope_u', 'parapet_height_m', 'wall_length_side',
        'wall_length_front', 'wall_fenesteration_per_front', 'roof_shape_u',
        'wall_substrate_u', 'retrofit_present_u'
    ]) & set(df_qs.columns) & set(df_nash.columns))
    
    # ... (simplified reconstruction of preprocessing)
    df = pd.concat([df_qs[common_cols], df_nash[common_cols]], axis=0)
    
    # Encode
    cat_cols = ['roof_shape_u', 'wall_substrate_u', 'retrofit_present_u']
    existing_cat = [c for c in cat_cols if c in df.columns]
    
    df_encoded = pd.get_dummies(df, columns=existing_cat, dummy_na=True)
    
    # Feature Cols
    feature_cols = [c for c in df_encoded.columns if c not in ['damage_indicator_u', 'tornado_EF']]
    feature_cols.append('T (tornado_EF normalized)') # Append T which is added last
    
    print("ZENN Input Features List:")
    for i, f in enumerate(feature_cols):
        print(f"{i}: {f}")

if __name__ == "__main__":
    list_features()

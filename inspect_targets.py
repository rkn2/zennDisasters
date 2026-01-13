
import pandas as pd
import numpy as np

def inspect_targets():
    # Load Data (Same logic as preprocess)
    csv_path = '/Users/rebeccanapolitano/antigravityProjects/featImp/updatedData/Revised_QuadState_Tornado_DataInput_pub - Copy_120525.csv'
    xlsx_path = '/Users/rebeccanapolitano/antigravityProjects/featImp/Nashville_Tornado_DataInput_Final_110725.xlsx'
    
    df_qs = pd.read_csv(csv_path)
    df_nash = pd.read_excel(xlsx_path)
    
    # Keep relevant cols
    cols = ['damage_indicator_u']
    
    df_qs = df_qs[cols]
    df_nash = df_nash[cols]
    
    df = pd.concat([df_qs, df_nash], axis=0, ignore_index=True)
    
    df['damage_indicator_u'] = pd.to_numeric(df['damage_indicator_u'], errors='coerce')
    df = df.dropna(subset=['damage_indicator_u'])
    
    print("Full Scale Distribution (damage_indicator_u):")
    print(df['damage_indicator_u'].value_counts().sort_index())
    
    # Save to file for tool check
    with open("target_dist.txt", "w") as f:
        f.write(str(df['damage_indicator_u'].value_counts().sort_index()))

if __name__ == "__main__":
    inspect_targets()

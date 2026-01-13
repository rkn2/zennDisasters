
import pandas as pd
import os

def inspect():
    base_path = '/Users/rebeccanapolitano/antigravityProjects/featImp'
    
    files = [
        'Nashville_Tornado_DataInput_Final_110725.xlsx',
        'updatedData/Revised_QuadState_Tornado_DataInput_pub - Copy_120525.csv',
        'QuadState_Tornado_DataInputv2.csv'
    ]
    
    with open('data_info.txt', 'w') as f:
        for file in files:
            path = os.path.join(base_path, file)
            if not os.path.exists(path):
                f.write(f"File not found: {path}\n")
                continue
                
            f.write(f"\n--- {file} ---\n")
            try:
                if file.endswith('.xlsx'):
                    df = pd.read_excel(path, nrows=5)
                else:
                    df = pd.read_csv(path, nrows=5)
                
                f.write("Columns:\n")
                for c in df.columns:
                    f.write(f"  {c}\n")
                f.write("\nFirst row:\n")
                f.write(str(df.iloc[0].to_dict()) + "\n")
            except Exception as e:
                f.write(f"Error reading: {e}\n")

if __name__ == "__main__":
    inspect()

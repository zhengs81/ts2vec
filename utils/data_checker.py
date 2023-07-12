import numpy as np
import pandas as pd
import glob
import math, os


import pandas as pd

def validate_files(file_paths):
    """
    This function checks all train files for their format
    """
    invalid_files = []
    
    for file_path in file_paths:
        print(invalid_files)
        try:
            df = pd.read_csv(file_path)
            
            # Check number of columns
            if len(df.columns) != 2:
                invalid_files.append(file_path)
                continue
            
            # if no header 
            if 'timestamp' not in df.columns or 'value' not in df.columns:
                invalid_files.append(file_path)
                continue
            
            # Check timestamp column format
            if df['timestamp'].apply(lambda x: isinstance(x, str)).all():
                df['timestamp'] = df['timestamp'].str.replace(r'\D', '', regex=True)
                if df['timestamp'].apply(lambda x: len(x) not in [10, 13]).any():
                    invalid_files.append(file_path)
                    continue
            
            # Check if timestamp column contains datetime values
            if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                invalid_files.append(file_path)
                continue
        
        except Exception as e:
            invalid_files.append(file_path)
            continue
    
    return invalid_files



def read_files_in_folder(folder_path):
    """
    This function returns path of all files under folder_path, at any sub-folder depth
    """
    res = []
    # Iterate over all files in the folder
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name != '.DS_Store' and not file_name.endswith(".zip"):
                res.append(os.path.join(root, file_name))
    return res

def main(path):
    all_paths = read_files_in_folder(path)
    return validate_files(all_paths)


if __name__ == '__main__':
    print(main("../data"))
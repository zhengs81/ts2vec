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
        # print(invalid_files)
        try:
            df = pd.read_csv(file_path)
            
            # Check number of columns
            if len(df.columns) != 2:
                invalid_files.append(file_path)
                continue
            
            # if no header, handled in load_data.py
            if 'timestamp' not in df.columns or 'value' not in df.columns:
                invalid_files.append(file_path)
                continue
            
            # Check timestamp column format, 13 or 10 digits long, start with 1
            if df['timeseries'].apply(lambda x: str(x).isdigit() and len(str(x)) in [10, 13] and str(x).startswith('1')).all():
                invalid_files.append(file_path)
                continue
                
            # Check value is either int or float
            if df['value'].apply(lambda x: isinstance(x, (int, float))).all():
                invalid_files.append(file_path)
                print(file_path)
                continue
            
            # Check if timestamp column contains datetime values
            if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                invalid_files.append(file_path)
                print(file_path)
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

def remove_ds_store(folder_path):
    # Recursively search for and remove .DS_Store files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == ".DS_Store":
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print("Removed:", file_path)
                except Exception as e:
                    print("An error occurred while removing", file_path, ":", str(e))

if __name__ == '__main__':
    # main("../data/train")
    remove_ds_store("../data/train")

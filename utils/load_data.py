import numpy as np
import pandas as pd
import glob
import math, os

def _load_files(selected_files):
    results = []
    for f in selected_files:
        data = pd.read_csv(f)

        # if no header 
        if 'timestamp' not in data.columns or 'value' not in data.columns:
            # Add headers if they are missing
            if 'timestamp' not in data.columns:
                data.rename(columns={data.columns[0]: 'timestamp'}, inplace=True)
            if 'value' not in data.columns:
                data.rename(columns={data.columns[1]: 'value'}, inplace=True)

        datetimes = None

        if len(str(data['timestamp'][0])) == 10:
            # 时间戳为10位时间戳
            datetimes = pd.to_datetime(data.timestamp, unit='s')
        elif len(str(data['timestamp'][0])) == 13:
            # 时间戳为13位时间戳
            datetimes = pd.to_datetime(data.timestamp, unit='ms')
        else:
            raise ValueError('Timestamp in wrong format')

        # 获取index为datetimes的pandas.DataFrame
        data['timestamp'] = datetimes
        # 找全时间戳并填充
        start = datetimes.min()
        end = datetimes.max()
        full_idx = pd.date_range(start, end, freq="min")
        filled_data = data.set_index('timestamp').reindex(full_idx)
        
        # 将缺失值进行插值、前后向填充
        filled_data = filled_data.interpolate().ffill().bfill()

        # 转换为numpy对象
        timeseries = np.array(filled_data['value'])  
        # normalize
        mean = np.mean(timeseries)
        var = np.var(timeseries)
        timeseries = (timeseries - mean) / math.sqrt(var)
        
        results.append(timeseries)

    return results

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

def load_train_data(num_datasets, path):
    """
    load all files under path, if num_datasets >= 0, then only fetch num_datasets files.
    If num_datasets < 0, load all files
    """
    files = read_files_in_folder(path)
    print("number of train files: ", len(files))

    selected_files = files[: min(num_datasets, len(files))] if num_datasets >= 0 else files

    return _load_files(selected_files)

def load_test_data(num_datasets):
    files = glob.glob("data/test/*") # 注意，如果在validate里跑，要改成../data .....
    print("number of test files: ",len(files))

    selected_files = files[: min(num_datasets, len(files))]

    return _load_files(selected_files)


if __name__ == '__main__':
    load_train_data(5, 'data')
    # print(read_files_in_folder("../data"))
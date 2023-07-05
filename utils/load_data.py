import numpy as np
import pandas as pd
import glob


def _load_files(selected_files):
    results = []
    for f in selected_files:
        data = pd.read_csv(f)
        datetimes = None
        if len(str(data['timestamp'][0])) == 10:
            # 时间戳为10位时间戳
            datetimes = pd.to_datetime(data.timestamp, unit='s')
        elif len(str(data['timestamp'][0])) == 13:
            # 时间戳为13位时间戳
            datetimes = pd.to_datetime(data.timestamp, unit='ms')

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
        results.append(timeseries)

    return results


def load_data(num_datasets):
    files = glob.glob("data/*.csv")

    selected_files = files[: min(num_datasets, len(files))]

    return _load_files(selected_files)


if __name__ == '__main__':
    load_data(1)

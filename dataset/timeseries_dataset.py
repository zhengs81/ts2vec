from torch.utils.data import Dataset
from bisect import bisect_left
import torch



# 自定义数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, stride):
        # data is an list of array, with each array being data from one file
        self.data = data
        total = 0
        self.suffix_len = []
        for d in self.data:
            total += len(d)
            self.suffix_len.append(total)
        self.seq_length = seq_length
        self.stride = stride

    def __len__(self):
        lengths = [len(l) for l in self.data]
        return (sum(lengths) - self.seq_length) // self.stride + 1

    def __getitem__(self, index):
        strided_idx = index * self.stride # idx in the whole dataset
        data_idx = bisect_left(self.suffix_len, strided_idx) # which dataset to use
        assert data_idx < len(self.data)
        dataset_len = len(self.data[data_idx]) # length of the chosen dataset
        prev_len = self.suffix_len[data_idx - 1] if data_idx else 0 # total length of previous datasets
        curr_idx = strided_idx - prev_len # idx in chosen dataset
        offset = curr_idx - min(dataset_len - self.seq_length, curr_idx)
        seq = self.data[data_idx][curr_idx - offset: curr_idx + self.seq_length - offset]

        assert len(seq) == self.seq_length
        return torch.tensor(seq, dtype=torch.float32)
    


class TimeSeriesTestDataset(Dataset):
    """
    Dataset used for evaluation, such as knn_plot. The difference is that it will also
    return the meta info for each sequence
    """
    def __init__(self, data, seq_length, stride):
        # data is an list of array, with each array being data from one file
        self.data = data
        total = 0
        self.suffix_len = []
        for d in self.data:
            total += len(d)
            self.suffix_len.append(total)
        self.seq_length = seq_length
        self.stride = stride

    def __len__(self):
        lengths = [len(l) for l in self.data]
        return (sum(lengths) - self.seq_length) // self.stride + 1

    def __getitem__(self, index):
        strided_idx = index * self.stride # idx in the whole dataset
        data_idx = bisect_left(self.suffix_len, strided_idx) # which dataset to use
        assert data_idx < len(self.data)
        dataset_len = len(self.data[data_idx]) # length of the chosen dataset
        prev_len = self.suffix_len[data_idx - 1] if data_idx else 0 # total length of previous datasets
        curr_idx = strided_idx - prev_len # idx in chosen dataset
        offset = curr_idx - min(dataset_len - self.seq_length, curr_idx)
        seq = self.data[data_idx][curr_idx - offset: curr_idx + self.seq_length - offset]

        return (data_idx, curr_idx - offset), torch.tensor(seq, dtype=torch.float32)
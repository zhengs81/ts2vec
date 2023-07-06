import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from TCN import TemporalConvNet


class Pretrained(nn.Module):
    def __init__(self, args, channels, temporal_unit = 0, device='cpu'):
        super(Pretrained, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.temporal_unit = temporal_unit
        self.net = Encoder(channels, args.input_dim, args, args.hidden_dim)
        self.device = device

    def forward(self, x):
        ts_l = x.size(1)
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)  # 随机生成截取的序列长度，范围是从2的(self.temporal_unit + 1)次方到整个序列长度ts_l+1
        crop_left = np.random.randint(ts_l - crop_l + 1)  # 随机生成截取序列的左边界crop_left
        crop_right = crop_left + crop_l  # 计算截取序列的右边界crop_right
        crop_eleft = np.random.randint(crop_left + 1)  # 随机生成裁剪序列的左扩展边界crop_eleft
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)  # 随机生成裁剪序列的右扩展边界crop_eright
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
        # 随机生成偏移量，在[-crop_eleft, ts_l - crop_eright + 1]范围内生成大小为batch_size的一维数组
        enh1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
        enh2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
        out1 = self.net(enh1)
        out2 = self.net(enh2)
        out1 = out1[:, -crop_l:]
        out2 = out2[:, -crop_l:]

        return out1, out2
def take_per_row(A, indx, num_elem):
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]

class Encoder(nn.Module):
    def __init__(self, channels, input_dims, args, hidden_dims=64):
        super().__init__()
        self.hidden_dims = hidden_dims
        # self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(input_dims, hidden_dims)
        self.network = TemporalConvNet(input_dims, hidden_dims, channels)

    def forward(self, x):  # x: B x T x hidd_dims
        # x = self.fc(x)
        out = self.network(x.permute(0, 2, 1)) # fc去掉
        return out.permute(0, 2, 1)
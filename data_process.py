import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from TCN import TemporalConvNet
from utils.load_data import load_data

# 自定义预训练模型（TCN+实例对比）
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


def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d


def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss

# 自定义数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, stride):
        self.data = data
        self.seq_length = seq_length
        self.stride = stride

    def __len__(self):
        # return (len(self.data) - self.seq_length)//self.stride + 1
        # return len(self.data) - self.seq_length
        return (len(self.data) - self.seq_length) // self.stride + 1

    def __getitem__(self, index):
        strided_idx = index * self.stride
        seq = self.data[strided_idx: strided_idx + self.seq_length]
        # 正样本构建
        # shift_size = np.random.randint(-self.stride // 2, self.stride // 2)
        # # 注意下shift_size需要排除掉引起`OutOfBoundError`的取值
        # postive_contrast = self.data[strided_idx + shift_size, strided_idx + self.seq_length].values

        return torch.tensor(seq, dtype=torch.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_length', default=1440, type=int, help='The length of the time series')
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--input_dim', default=1, type=int, help='input dimension')
    parser.add_argument('--hidden_dim', default=64, type=int, help='hidden dimension')
    parser.add_argument('--num_epochs', default=100, type=int, help='training iteration')
    parser.add_argument('--batch_size', default=8, type=int, help='number of example per batch')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--num_dataset', default=1, type=int, help='number of dataset to process')
    parser.add_argument('--device', default='cpu', type=str)
    args = parser.parse_args()
    
    args.device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    timeseries = load_data(args.num_dataset)[1]  # 加载数据

    dataset = TimeSeriesDataset(timeseries, args.seq_length, args.stride) 
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    channels = [4, 8]
    model = Pretrained(args, channels).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        sum_loss = 0
        for batch in dataloader: 
            # input is of shape [batch_size, seq_length, num_features=1], [8, 1440, 1]
            inputs = batch.unsqueeze(-1)
            inputs = inputs.to(args.device)
            outputs1, outputs2 = model(inputs)
            loss = hierarchical_contrastive_loss(outputs1, outputs2)  # 对比损失
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss
            print('Epoch:{}, batch_loss={}'.format(epoch, loss))
        print('Epoch:{}, loss={}'.format(epoch, sum_loss))

    #  保存模型
    torch.save(model.state_dict(), 'pretrained_model.pt')

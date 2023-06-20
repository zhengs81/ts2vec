import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 自定义预训练模型（TCN+实例对比）
class Pretrained(nn.Module):
    def __init__(self):
        super(Pretrained, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        return x


# 自定义数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.daata) - self.seq_length

    def __getitem__(self, index):
        seq = self.data[index:index + self.seq_length].values
        return torch.tensor(seq, dtype=torch.float32)


data = pd.read_csv('farseer_cases/交易量指标/10.0.210.11.tps.csv')
batch_size = 32
# 按天划分时间序列
seq_length = 1440  # 每个时间序列的长度为一天（1440分钟）
dataset = TimeSeriesDataset(data['value'], seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_epochs = 100

# 初始化模型和优化器
model = Pretrained()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch in dataloader:
        inputs = batch.unsqueeze(1)  # 添加通道维度
        outputs = model(inputs)
        loss = ...  # 损失函数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 保存模型
torch.save(model.state_dict(), 'pretrained_model.pth')

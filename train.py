import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.load_data import load_train_data
from model.pretrained import Pretrained
from loss.hierarchical_contrastive_loss import hierarchical_contrastive_loss
from dataset.timeseries_dataset import TimeSeriesDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_length', default=20, type=int, help='The length of the time series')
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--input_dim', default=1, type=int, help='input dimension')
    parser.add_argument('--hidden_dim', default=64, type=int, help='hidden dimension')
    parser.add_argument('--num_epochs', default=100, type=int, help='training iteration')
    parser.add_argument('--batch_size', default=8, type=int, help='number of example per batch')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--num_dataset', default=-1, type=int, help='number of dataset to process, -1 for all')
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--train_dir', default='data/train', type=str, help='path to dir where the train data is stored')
    args = parser.parse_args()
    
    args.device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    timeseries = load_train_data(args.num_dataset, args.train_dir)  # 加载数据

    dataset = TimeSeriesDataset(timeseries, args.seq_length, args.stride) 
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    channels = [2**i for i in range(1, 11)]
    model = Pretrained(args, channels).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    torch.save(args, 'saved_models/pretrained_model_params.pt')

    print("Start Training ...")
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
    torch.save(model.state_dict(), 'saved_models/pretrained_model.pt')
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
    parser.add_argument('--num_dataset', default=-1, type=int, help='number of dataset to process, -1 to all')
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
temp = tensor([[[ 1.1459],         [ 1.1569],         [ 1.1680],         [ 1.1791],         [ 1.5835],         [ 1.9880],         [ 2.3924],         [ 2.7968],         [ 3.2013],         [ 3.1598],         [ 3.1182],         [ 3.0767],         [ 3.0352],         [ 2.9937],         [ 2.7977],         [ 2.6017],         [ 2.4056],         [ 2.2096],         [ 2.0136],         [ 1.8718]],        [[ 0.0000],         [ 0.0000],         [ 0.0000],         [ 0.0000],         [ 0.0000],         [ 0.0000],         [ 0.0000],         [ 0.0000],         [ 0.0000],         [ 0.0000],         [ 0.0000],         [ 0.0000],         [ 0.0000],         [ 0.0000],         [ 0.0000],         [ 0.0000],         [ 0.0000],         [ 0.0000],         [ 0.0000],         [ 0.0000]],        [[ 0.6443],         [ 0.5788],         [ 0.7753],         [ 0.6006],         [ 0.7535],         [ 0.6880],         [ 0.7006],         [ 0.9074],         [ 0.5133],         [ 0.6317],         [ 0.4963],         [ 0.5934],         [ 0.7098],         [ 0.5570],         [ 0.4915],         [ 0.6546],         [ 0.6443],         [ 0.5133],         [ 0.6443],         [ 0.6006]],        [[    nan],         [    nan],         [    nan],         [    nan],         [    nan],         [    nan],         [    nan],         [    nan],         [    nan],         [    nan],         [    nan],         [    nan],         [    nan],         [    nan],         [    nan],         [    nan],         [    nan],         [    nan],         [    nan],         [    nan]],        [[ 0.6414],         [ 0.6414],         [ 0.6414],         [ 0.6414],         [ 0.6414],         [ 0.6414],         [ 0.6414],         [ 0.6414],         [ 0.6414],         [ 0.6414],         [ 0.6414],         [ 0.6414],         [ 0.6414],         [ 0.6414],         [ 0.6414],         [ 0.6414],         [ 0.6414],         [ 0.6414],         [ 0.6414],         [ 0.6414]],        [[ 3.9831],         [ 4.0214],         [ 4.0596],         [ 4.0979],         [ 4.1361],         [ 4.0903],         [ 4.0444],         [ 3.9985],         [ 3.9526],         [ 3.9067],         [ 4.4200],         [ 4.9333],         [ 5.4466],         [ 5.9598],         [ 6.4731],         [ 6.5681],         [ 6.6631],         [ 6.7581],         [ 6.8531],         [ 6.9482]],        [[-0.0423],         [-0.0377],         [-0.0322],         [-0.0405],         [-0.0366],         [-0.0430],         [-0.0462],         [-0.0477],         [-0.0446],         [-0.0457],         [-0.0452],         [-0.0292],         [-0.0451],         [-0.0421],         [-0.0398],         [-0.0451],         [-0.0474],         [-0.0389],         [-0.0346],         [-0.0389]],        [[ 2.0813],         [ 0.7362],         [ 0.7362],         [ 2.0813],         [ 0.7362],         [-0.6090],         [-0.6090],         [ 3.4264],         [ 0.7362],         [-0.6090],         [-0.6090],         [-0.6090],         [-0.6090],         [ 0.7362],         [-0.6090],         [-0.6090],         [-0.6090],         [-0.6090],         [-0.6090],         [-0.6090]]])
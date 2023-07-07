import os, sys, torch, argparse
from argparse import Namespace
from torch.utils.data import DataLoader
from argparse import Namespace
# Set path to SHAPE
current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current))
from model.pretrained import Pretrained
from utils.load_data import load_test_data
from dataset.timeseries_dataset import TimeSeriesTestDataset

def load_model(path, gpu, params):
    model = Pretrained(params, [4,8])
    if gpu:
        model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage.cuda(gpu)))
    else:
        model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    model.eval()
    return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=1, type=int, help='number of files to test')
    parser.add_argument('--model', default="saved_models/pretrained_model.pt", type=str, help='The path to model parameters')
    parser.add_argument('--gpu', default=0, type=int, help='GPU index to use, if CUDA is enabled')
    parser.add_argument('--stride', default=72, type=int)
    parser.add_argument('--seq_length', default=144, type=int, help='GPU index to use, if CUDA is enabled')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of batches used for splitting the test data to avoid out of memory errors when using CUDA.')
    args = parser.parse_args()

    args.device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    data = load_test_data(args.data)
    dataset = TimeSeriesTestDataset(data, args.seq_length, args.stride) 
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    params = torch.load("saved_models/pretrained_model_param.pt")
    model = load_model(args.model, args.gpu, params)

    for metadata, batch in dataloader: # batch [128, 144]
        # metadata is list of 2 tensors
        input = batch.unsqueeze(-1).to(args.device)
        output = model.net(input) # output [128, 64, 8]



    encoded_rep = model.encode_window(data, args.window_len, args.batch_size)
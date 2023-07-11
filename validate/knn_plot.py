import os, sys, torch, argparse
from torch.utils.data import DataLoader
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
# Set path to SHAPE
current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current))
from model.pretrained import Pretrained
from utils.load_data import load_test_data
from dataset.timeseries_dataset import TimeSeriesTestDataset

def load_model(path, gpu, params):
    model = Pretrained(params, [2**i for i in range(1, 11)])
    if gpu:
        model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage.cuda(gpu)))
    else:
        model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    model.eval()
    return model

def avgpool_timestamp_output(output):
    """
    Given output from encoder, this function performs average pool to squeeze T dim
    output: [B, T, hidden]
    """
    return np.average(output, axis=-1)

def maxpool_timestamp_output(output):
    """
    Given output from encoder, this function performs average pool to squeeze T dim
    output: [B, T, hidden]
    """
    return np.amax(output, axis=-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=1, type=int, help='number of files to test')
    parser.add_argument('--model', default="../saved_models/pretrained_model.pt", type=str, help='The path to model parameters')
    parser.add_argument('--gpu', default=0, type=int, help='GPU index to use, if CUDA is enabled')
    parser.add_argument('--stride', default=72, type=int)
    parser.add_argument('--seq_length', default=144, type=int, help='GPU index to use, if CUDA is enabled')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of batches used for splitting the test data to avoid out of memory errors when using CUDA.')
    parser.add_argument('--num_neighbours', default=5, type=int, help='number of closet neighbours to find, including itself')
    args = parser.parse_args()

    args.device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    data = load_test_data(args.data)
    dataset = TimeSeriesTestDataset(data, args.seq_length, args.stride) 
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    params = torch.load("../saved_models/pretrained_model_params.pt")
    model = load_model(args.model, args.gpu, params)

    dataset_index = np.array([]) # index of the dataset at where the query is located
    slice_index = np.array([]) # index within the dataset, query length is args.seq_length
    encoded_vec = np.zeros((1, args.seq_length)) # to store encoded vec, shape of [total_batch, hidden_dim]

    with torch.no_grad():
        for metadata, batch in dataloader: # batch [128, 144, 1]
            # metadata is list of 2 tensors with each tensor to be B(128) length
            dataset_index = np.append(dataset_index, metadata[0]) # notice after append, still 1D array
            slice_index = np.append(slice_index, metadata[1])
            input = batch.unsqueeze(-1).to(args.device)
            output = model.net(input) # output [B, T, hidden]
            squeezed_out = avgpool_timestamp_output(output) # [B, T], or maxpool_timestamp_output
            encoded_vec = np.append(encoded_vec, squeezed_out, axis=0)

    encoded_vec = encoded_vec[1:] # remove the first zero-row
    
    # KD tree 
    tree = KDTree(encoded_vec)
    random_test_idx = np.random.randint(0, encoded_vec.shape[0]) # which query to test
    dist, ind = tree.query(encoded_vec[random_test_idx: random_test_idx+1], k = args.num_neighbours)
    ind = ind[0]
    
    neighbors_seq = []

    for i in ind:
        d, s = int(dataset_index[i]), int(slice_index[i])
        neighbors_seq.append(data[d][s: s+args.seq_length])

    # now neighbors_seq is a length k list, with each element being length args.seq_length array

    # plot
    space = int(0.3 * args.seq_length)

    x_axis = [list(range(i * (args.seq_length + space), i * (args.seq_length + space) + args.seq_length)) for i in range(args.num_neighbours)]

    fig, ax = plt.subplots()

    for i in range(args.num_neighbours):
        ax.plot(x_axis[i], neighbors_seq[i], label='Test Sequence' if not i else 'Closet neighbour # '+str(i))

    # Set labels and legend
    ax.set_xlabel("tested sequence with its " + str(args.num_neighbours) + " closet neighbours")
    ax.set_ylabel('value')
    ax.legend()

    # Show the plot
    plt.show()
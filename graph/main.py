import argparse
import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree

from graph import run
from conv import GMMConv

parser = argparse.ArgumentParser(description='Cora citation network')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--device_idx', type=int, default=1)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--early_stopping', type=int, default=50)
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--kernel_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
args = parser.parse_args()
print(args)

def transform(data):
    row, col = data.edge_index
    deg = degree(col, data.num_nodes)
    data.edge_attr = torch.stack(
        [1 / torch.sqrt(deg[row]), 1 / torch.sqrt(deg[col])], dim=-1)
    return data


class MoNet(torch.nn.Module):
    def __init__(self, dataset):
        super(MoNet, self).__init__()
        self.conv1 = GMMConv(dataset.num_features,
                             args.hidden,
                             dim=2,
                             kernel_size=args.kernel_size)
        self.conv2 = GMMConv(args.hidden,
                             dataset.num_classes,
                             dim=2,
                             kernel_size=args.kernel_size)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda', args.device_idx)
args.data_fp = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                        args.dataset)
dataset = Planetoid(args.data_fp, args.dataset)
dataset.transform = transform
run(dataset, MoNet(dataset), args.runs, args.epochs, args.lr,
    args.weight_decay, args.early_stopping, device)

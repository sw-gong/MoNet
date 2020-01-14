import argparse
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch_geometric.datasets import FAUST
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

from conv import GMMConv
from manifolds import run

parser = argparse.ArgumentParser(description='shape correspondence')
parser.add_argument('--dataset', type=str, default='FAUST')
parser.add_argument('--device_idx', type=int, default=4)
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--kernel_size', type=int, default=10)
parser.add_argument('--lr', type=float, default=3e-3)
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

args.data_fp = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                        args.dataset)
device = torch.device('cuda', args.device_idx)
torch.set_num_threads(args.n_threads)

# deterministic
torch.manual_seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True


class Pre_Transform(object):
    def __call__(self, data):
        data.x = data.pos
        data = T.FaceToEdge()(data)

        return data


train_dataset = FAUST(args.data_fp,
                      True,
                      transform=T.Cartesian(),
                      pre_transform=Pre_Transform())
test_dataset = FAUST(args.data_fp,
                     False,
                     transform=T.Cartesian(),
                     pre_transform=Pre_Transform())
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)
d = train_dataset[0]
target = torch.arange(d.num_nodes, dtype=torch.long, device=device)
print(d)


class MoNet(nn.Module):
    def __init__(self, in_channels, num_classes, kernel_size):
        super(MoNet, self).__init__()

        self.fc0 = nn.Linear(in_channels, 16)
        self.conv1 = GMMConv(16, 32, dim=3, kernel_size=kernel_size)
        self.conv2 = GMMConv(32, 64, dim=3, kernel_size=kernel_size)
        self.conv3 = GMMConv(64, 128, dim=3, kernel_size=kernel_size)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        nn.init.xavier_uniform_(self.fc0.weight, gain=1)
        nn.init.xavier_uniform_(self.fc1.weight, gain=1)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        nn.init.constant_(self.fc0.bias, 0)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.fc0(x))
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = F.elu(self.conv3(x, edge_index, edge_attr))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = MoNet(d.num_features, d.num_nodes, args.kernel_size).to(device)
print(model)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      args.decay_step,
                                      gamma=args.lr_decay)

run(model, train_loader, test_loader, target, d.num_nodes, args.epochs,
    optimizer, scheduler, device)

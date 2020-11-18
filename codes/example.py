import os.path as osp
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', default=True, action='store_true',
                    help='Use GDC preprocessing.')
parser.add_argument('--data', default='Cora')
parser.add_argument('--model', default='Net')
parser.add_argument('--ppr', default=0.15)
parser.add_argument('--topk', default=5)

args = parser.parse_args()


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, model_dim, d_hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(model_dim, d_hidden)
        self.w_2 = nn.Linear(d_hidden, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.init()

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

    def init(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.w_1.weight, gain=gain)
        nn.init.xavier_normal_(self.w_2.weight, gain=gain)




def main(args):
    dataset = args.data
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]

    if args.use_gdc:
        gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                    normalization_out='col',
                    diffusion_kwargs=dict(method='ppr', alpha=args.ppr), ## orig 0.05
                    sparsification_kwargs=dict(method='topk', k=args.topk,
                                               dim=0), exact=True)
        data = gdc(data)

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                                 normalize=not args.use_gdc)
            self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                                 normalize=not args.use_gdc)

        def forward(self):
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            return F.log_softmax(x, dim=1)

    class NetLayerNorm_FF(torch.nn.Module):
        def __init__(self):
            super(NetLayerNorm_FF, self).__init__()
            self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                                 normalize=not args.use_gdc)
            self.layer_norm_1 = torch.nn.LayerNorm(16)
            self.ff_layer = PositionwiseFeedForward(model_dim=16, d_hidden=8 * 16)
            self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                                 normalize=not args.use_gdc)

        def forward(self):
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            x = self.conv1(x, edge_index, edge_weight)
            norm_x = self.layer_norm_1(x)
            norm_x = F.dropout(norm_x, training=self.training)
            x = self.ff_layer(x + norm_x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            return F.log_softmax(x, dim=1)

    class DeepNet(torch.nn.Module):
        def __init__(self, layers=3):
            super(DeepNet, self).__init__()
            self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                                 normalize=not args.use_gdc)
            self.multi_conv_layers = nn.ModuleList()
            self.multi_conv_layers.append(self.conv1)
            for i in range(1, layers):
                layer_i = GCNConv(16, 16, cached=True,
                                  normalize=not args.use_gdc)
                self.multi_conv_layers.append(layer_i)
            self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                                 normalize=not args.use_gdc)

        def forward(self):
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            for layer_i in self.multi_conv_layers:
                x = F.relu(layer_i(x, edge_index, edge_weight)) + x
                x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            return F.log_softmax(x, dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == 'Net':
        model, data = Net().to(device), data.to(device)
    elif args.model == 'NetFF':
        model, data = NetLayerNorm_FF().to(device), data.to(device)
    else:
        model, data = DeepNet().to(device), data.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=0.01)  # Only perform weight-decay on first convolution.


    def train():
        model.train()
        optimizer.zero_grad()
        F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
        optimizer.step()


    @torch.no_grad()
    def test():
        model.eval()
        logits, accs = model(), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs


    best_val_acc = test_acc = 0
    for epoch in range(1, 401):
        train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))
    print('model {}: data: {}: test_acc: {}'.format(args.model, args.data, test_acc))

main(args=args)
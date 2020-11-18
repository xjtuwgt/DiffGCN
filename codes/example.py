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
parser.add_argument('--ppr', default=0.15, type=float) # 0.05
parser.add_argument('--topk', default=5, type=int)  #128
parser.add_argument('--hid_dim', default=16, type=int)  #16
parser.add_argument('--weight_decay', default=1e-4, type=float) #5e-4
parser.add_argument('--lr', default=0.01, type=float) #5e-4

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
            self.conv1 = GCNConv(dataset.num_features, args.hid_dim, cached=True,
                                 normalize=not args.use_gdc)
            self.conv2 = GCNConv(args.hid_dim, dataset.num_classes, cached=True,
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
            self.conv1 = GCNConv(dataset.num_features, args.hid_dim, cached=True,
                                 normalize=not args.use_gdc)
            if dataset.num_features != args.hid_dim:
                self.res_fc = nn.Linear(dataset.num_features, args.hid_dim, bias=False)
            else:
                self.res_fc = None
            self.layer_norm_1 = torch.nn.LayerNorm(args.hid_dim)
            self.ff_layer = PositionwiseFeedForward(model_dim=args.hid_dim, d_hidden=4 * args.hid_dim)
            self.conv2 = GCNConv(args.hid_dim, dataset.num_classes, cached=True,
                                 normalize=not args.use_gdc)

        def forward(self):
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            convx = self.conv1(x, edge_index, edge_weight)
            if self.res_fc is not None:
                res_x = self.res_fc(x)
            else:
                res_x = x
            convx = convx + res_x
            norm_x = self.layer_norm_1(convx)
            norm_x = F.dropout(norm_x, training=self.training)

            x_ff = self.ff_layer(norm_x)
            x_ff = F.dropout(x_ff, training=self.training)
            x = norm_x + x_ff
            x = self.conv2(x, edge_index, edge_weight)
            return F.log_softmax(x, dim=1)

    class DeepNet(torch.nn.Module):
        def __init__(self, layers=3):
            super(DeepNet, self).__init__()
            self.conv1 = GCNConv(dataset.num_features, args.hid_dim, cached=True,
                                 normalize=not args.use_gdc)
            self.multi_conv_layers = nn.ModuleList()
            if dataset.num_features != args.hid_dim:
                self.res_fc = nn.Linear(dataset.num_features, args.hid_dim, bias=False)
            else:
                self.res_fc = None

            for i in range(1, layers):
                layer_i = GCNConv(args.hid_dim, args.hid_dim, cached=True,
                                  normalize=not args.use_gdc)
                self.multi_conv_layers.append(layer_i)
            self.conv2 = GCNConv(args.hid_dim, dataset.num_classes, cached=True,
                                 normalize=not args.use_gdc)

        def forward(self):
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            if self.res_fc is not None:
                res_x = self.res_fc(x)
            x = x + res_x
            for layer_i in self.multi_conv_layers:
                x_temp = F.dropout(x, training=self.training)
                x = F.relu(layer_i(x_temp, edge_index, edge_weight)) + x
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
        dict(params=model.conv1.parameters(), weight_decay=args.weight_decay),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=args.lr)  # Only perform weight-decay on first convolution.


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
    for epoch in range(1, 201):
        train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))
    print('model {}: data: {}: test_acc: {}'.format(args.model, args.data, test_acc))
    return test_acc

def model_selection(args):
    data = 'Cora'
    model = 'NetFF'
    best_setting = None
    ppr_range = [0.05, 0.1, 0.15, 0.2]
    topk_range = [32, 64, 128]
    hid_dim_range = [16, 32, 64, 128, 256]
    lr_range = [0.01, 0.005]
    weight_decay_range = [1e-4, 5e-5]
    best_acc = 0
    for ppr in ppr_range:
        for topk in topk_range:
            for hid_dim in hid_dim_range:
                for lr in lr_range:
                    for weight_decay in weight_decay_range:
                        args.ppr = ppr
                        args.data = data
                        args.model = model
                        args.topk = topk
                        args.lr = lr
                        args.weight_decay = weight_decay
                        args.hid_dim = hid_dim
                        test_acc_i = main(args)
                        if best_acc < test_acc_i:
                            best_acc = test_acc_i
                            best_setting = [ppr, topk, hid_dim, lr, weight_decay]
                        print('*' * 75)
    print('Data: {} Model: {}Best acc = {}, best setting = {}'.format(data, model, best_acc, best_setting))


model_selection(args=args)
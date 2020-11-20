import os.path as osp
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa

import numpy as np
import os
import subprocess
from io import StringIO
import torch
import pandas as pd
import numpy as np
import random
import os




def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode("utf-8")),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory_free'] = gpu_df['memory.free'].apply(lambda x: float(x.rstrip(' [MiB]')))
    gpu_df['memory_used'] = gpu_df['memory.used'].apply(lambda x: float(x.rstrip(' [MiB]')))
    idx = gpu_df['memory_free'].argmax()
    used_memory = gpu_df.iloc[idx]['memory_used']
    print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx, used_memory

def set_free_cuda():
    free_gpu_id, used_memory = get_free_gpu()
    device = torch.device('cuda:'+str(free_gpu_id))
    torch.cuda.set_device(device=device)
    return [free_gpu_id], used_memory

def get_multi_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode("utf-8")),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory_free'] = gpu_df['memory.free'].apply(lambda x: float(x.rstrip(' [MiB]')))
    gpu_df['memory_used'] = gpu_df['memory.used'].apply(lambda x: float(x.rstrip(' [MiB]')))
    idx = gpu_df['memory_free'].argmax()
    used_memory = gpu_df.iloc[idx]['memory_used']
    free_idxs = []
    for idx, row in gpu_df.iterrows():
        if row['memory_used'] <= used_memory:
            free_idxs.append(idx)
    print('Returning GPU {} with smaller than {} free MiB'.format(free_idxs, gpu_df.iloc[idx]['memory.free']))
    return free_idxs, used_memory

def set_multi_free_cuda():
    free_gpu_ids, used_memory = get_multi_free_gpu()
    aa = []
    for i in free_gpu_ids:
        device = torch.device("cuda:{}".format(i))
        aa.append(torch.rand(1).to(device))  # a place holder
    return free_gpu_ids, used_memory

def gpu_setting(num_gpu=1):
    if num_gpu > 1:
        return set_multi_free_cuda()
    else:
        return set_free_cuda()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
available_device_count = torch.cuda.device_count()
if available_device_count > 0:
    print('GPU number is {}'.format(available_device_count))
    # ++++++++++++++++++++++++++++++++++
    device_ids, used_memory = gpu_setting(available_device_count)
    # ++++++++++++++++++++++++++++++++++
    device = torch.device("cuda:{}".format(device_ids[0]))

def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


import random
import torch
def random_split(N, train_dev_test_tuple, random_seed: int = 0):
    random.seed(random_seed)
    data_ids = [i for i in range(N)]
    train_mask, valid_mask, test_mask = torch.zeros(N), torch.zeros(N), torch.zeros(N)
    train_size, valid_size, test_size = train_dev_test_tuple
    random.shuffle(data_ids)
    train_ids = data_ids[:train_size]
    valid_ids = data_ids[train_size:(train_size+valid_size)]
    test_ids = data_ids[(train_size+valid_size): (train_size + valid_size + test_size)]
    train_mask[train_ids] = 1
    valid_mask[valid_ids] = 1
    test_mask[test_ids] = 1
    assert train_mask.sum() == train_size
    train_mask = train_mask.type(torch.bool)
    valid_mask = valid_mask.type(torch.bool)
    test_mask = test_mask.type(torch.bool)
    return train_mask, valid_mask, test_mask

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', default=True, action='store_true',
                    help='Use GDC preprocessing.')
parser.add_argument('--data', default='Cora')
parser.add_argument('--model', default='NetFF')
parser.add_argument('--ppr', default=0.15, type=float) # 0.05
parser.add_argument('--topk', default=5, type=int)  #128
parser.add_argument('--hid_dim', default=16, type=int)  #16
parser.add_argument('--weight_decay', default=1e-4, type=float) #5e-4
parser.add_argument('--lr', default=0.01, type=float) #5e-4
parser.add_argument('--layers', default=3, type=int) #5e-4
parser.add_argument('--rand_seed', default=0, type=int) #5e-4
parser.add_argument('--shuffle', default=False, action='store_true',
                    help='Use GDC preprocessing.')


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
    set_seeds(args.rand_seed)
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
        if args.shuffle:
            print('Performing shuffle... over {} using model {}'.format(args.data, args.model))
            train_dev_test_tuple = (data.train_mask.sum().data.item(), data.val_mask.sum().data.item(), data.test_mask.sum().item())
            train_mask, val_mask, test_mask = random_split(N=data.train_mask.shape[0], train_dev_test_tuple=train_dev_test_tuple, random_seed=args.rand_seed)
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask
        else:
            print('Standard splitting...over {} using model {}'.format(args.data, args.model))

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
            self.layer_norm_2 = torch.nn.LayerNorm(args.hid_dim)
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
            norm_x = F.dropout(self.layer_norm_1(convx), training=self.training)
            norm_x = norm_x + res_x
            x_ff = self.ff_layer(norm_x)
            x_ff = F.dropout(x_ff, training=self.training)
            x = norm_x + x_ff
            x = self.layer_norm_2(x)
            x = self.conv2(x, edge_index, edge_weight)
            return F.log_softmax(x, dim=1)

    class DeepNet(torch.nn.Module):
        def __init__(self, layers=args.layers):
            super(DeepNet, self).__init__()
            self.conv1 = GCNConv(dataset.num_features, args.hid_dim, cached=True,
                                 normalize=not args.use_gdc)

            if dataset.num_features != args.hid_dim:
                self.res_fc = nn.Linear(dataset.num_features, args.hid_dim, bias=False)
            else:
                self.res_fc = None

            self.multi_conv_layers = nn.ModuleList()
            for i in range(2, layers):
                layer_i = GCNConv(args.hid_dim, args.hid_dim, cached=True,
                                  normalize=not args.use_gdc)
                self.multi_conv_layers.append(layer_i)
            self.conv2 = GCNConv(args.hid_dim, dataset.num_classes, cached=True,
                                 normalize=not args.use_gdc)

        def forward(self):
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            if self.res_fc is not None:
                res_x = self.res_fc(x)
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = x + res_x
            for layer_i in self.multi_conv_layers:
                x_temp = x
                x = F.relu(layer_i(x, edge_index, edge_weight))
                x = x + x_temp
                x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            return F.log_softmax(x, dim=1)



    if args.model == 'Net':
        model, data = Net().to(device), data.to(device)
    elif args.model == 'NetFF':
        model, data = NetLayerNorm_FF().to(device), data.to(device)
    elif args.model == 'Deep':
        model, data = DeepNet().to(device), data.to(device)
        print('Deep model layer number = {}'.format(args.layers))
    else:
        raise ValueError('model %s not supported' % args.model)

    print(model)
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
    for epoch in range(1, 301):
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
    best_setting = None
    ppr_range = [0.05, 0.1, 0.15, 0.2]
    topk_range = [32, 64, 128]
    hid_dim_range = [16, 64]
    lr_range = [0.01]
    weight_decay_range = [1e-4]
    best_acc = 0
    for ppr in ppr_range:
        for topk in topk_range:
            for hid_dim in hid_dim_range:
                for lr in lr_range:
                    for weight_decay in weight_decay_range:
                        args.ppr = ppr
                        args.topk = topk
                        args.lr = lr
                        args.weight_decay = weight_decay
                        args.hid_dim = hid_dim
                        test_acc_i = main(args)
                        if best_acc < test_acc_i:
                            best_acc = test_acc_i
                            best_setting = [ppr, topk, hid_dim, lr, weight_decay]
                        print('*' * 75)
    print('Data: {} Model: {} Best acc = {}, best setting = {}'.format(args.data, args.model, best_acc, best_setting))


model_selection(args=args)
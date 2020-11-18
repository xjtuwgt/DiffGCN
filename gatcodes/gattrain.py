import sys
import os


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import argparse
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

import numpy as np
import torch
import random
import dgl
import os
from dgl import DGLGraph

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
    dgl.random.seed(seed)


from gatcodes.gat import GAT
import argparse

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')

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

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)

def main(args):
    # load and preprocess dataset
    set_seeds(args.rand_seed)
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    if args.shuffle:
        print('Random splitting...')
        ####
        train_dev_test_tuple = (train_mask.sum().data.item(), val_mask.sum().data.item(), test_mask.sum().data.item())
        train_mask, val_mask, test_mask = random_split(N=g.num_nodes(), train_dev_test_tuple=train_dev_test_tuple, random_seed=args.rand_seed)
        ####
    else:
        print('standard splitting')

    num_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)
    print(model)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    best_val_acc = 0
    test_acc = 0
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        if args.fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
        else:
            val_acc = evaluate(model, features, labels, val_mask)
            if args.early_stop:
                if stopper.step(val_acc, model):
                    break

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            test_acc = evaluate(model, features, labels, test_mask)

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} TestAcc {:.4f}| ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, test_acc, n_edges / np.mean(dur) / 1000))

    print()
    if args.early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))
    acc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))
    print('Test Accuracy {:.4f}'.format(test_acc))
    return test_acc


def model_selection(args):
    num_hidden_range = [8, 16]
    in_drop_range = [0.2, 0.4, 0.6]
    att_drop_range = [0.2, 0.4, 0.6]
    lr_range = [0.005]
    best_acc = 0
    for num_hidden in num_hidden_range:
        for in_drop in in_drop_range:
            for att_drop in att_drop_range:
                for lr in lr_range:
                    args.num_hidden = num_hidden
                    args.in_drop = in_drop
                    args.attn_drop = att_drop
                    args.lr = lr
                    acc_i = main(args)
                    if best_acc < acc_i:
                        best_acc = acc_i
                print('*' * 75)
    print('Best accuracy for {} is {}'.format(args.dataset, best_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    # register_data_args(parser)
    parser.add_argument(
        "--dataset",
        type=str,
        default='cora',
        required=False,
        help=
        "The input dataset. Can be cora, citeseer, pubmed, syn(synthetic dataset) or reddit"
    )
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA preprocessing.')
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    parser.add_argument('--rand_seed', default=5, type=int,
                        help="skip re-evaluate the validation set")
    parser.add_argument('--shuffle', default=True, action='store_true',
                        help="random split")
    args = parser.parse_args()
    # print(args)

    # main(args)
    model_selection(args=args)
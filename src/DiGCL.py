
import os.path as osp
import argparse
import os, time
from datetime import datetime
import sys


import torch
from sklearn import metrics
import numpy as np
import networkx as nx
import pickle as pk
import random
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

from typing import List

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder



from torch_geometric_signed_directed.data import load_directed_real_data
from torch_geometric_signed_directed import node_class_split

from torch_geometric_signed_directed.utils import (
    cal_fast_appr, drop_feature) #, pred_digcl_node)
from torch_geometric_signed_directed.nn.directed import DiGCL
from utils.edge_data import in_out_degree

np.set_printoptions(threshold=sys.maxsize)

# select cuda device if available
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--log_root', type=str, default='../logs/', help='the path saving model.t7 and the training process')
parser.add_argument('--log_path', type=str, default='test', help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/', help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')
parser.add_argument('--dataset', type=str, default='WebKB/Cornell', help='data set selection')

parser.add_argument('--method_name', type=str, default='DiGCL', help='method name')
parser.add_argument('--activation', type=str, default = 'relu')
parser.add_argument('--tau', type=float, default =0.4)


parser.add_argument('--num_filter', type=int, default=2, help='num of filters')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--drop_feature_rate_1', type=float, default=0.3)
parser.add_argument('--drop_feature_rate_2', type=float, default=0.4)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--curr-type', type=str, default='log')

args = parser.parse_args()


save_name = args.method_name + 'lr' + str(int(args.lr*1000)) + 'num_filters' + str(int(args.num_filter)) + '_curr-type_' + str(args.curr_type) + '_activation_' +str(args.activation) + '_tau_' + str(float(args.tau))
args.save_name = save_name


def acc(pred, label, mask):
    label = label.cpu()
    mask = mask.cpu()
    correct = int(pred.eq(label[mask]).sum().item())
    acc = correct / int(mask.sum())
    return acc


def pred_digcl_node(embeddings: torch.FloatTensor, y: torch.LongTensor, train_index: List[int], test_index: List[int] = None):
    """ Generate predictions from embeddings from the
    """
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    p = np.random.permutation(len(X))
    X, Y = X[p], Y[p]
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(bool)
    X = normalize(X, norm='l2')
    X_train = X[train_index]
    y_train = Y[train_index]

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)
    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=10,
                       verbose=1)
    clf.fit(X_train, y_train)
    y_pred = np.argmax(clf.predict(X), axis=1)

    if test_index is None:
        return y_pred
    else:
        return y_pred[test_index]

def train(X, edge_index,
          alpha_1, alpha_2,
          drop_feature1, drop_feature2, edge_weight):
    model.train()
    optimizer.zero_grad()

    edge_index_1, edge_weight_1 = cal_fast_appr(
        alpha_1, edge_index, X.shape[0], X.dtype, edge_weight=edge_weight)
    edge_index_2, edge_weight_2 = cal_fast_appr(
        alpha_2, edge_index, X.shape[0], X.dtype, edge_weight=edge_weight)

    x_1 = drop_feature(X, drop_feature1)
    x_2 = drop_feature(X, drop_feature2)
    z1 = model(x_1, edge_index_1, edge_weight_1)
    z2 = model(x_2, edge_index_2, edge_weight_2)
    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(X, y, edge_index, edge_in, in_weight, edge_out, out_weight, mask):
    model.eval()
    with torch.no_grad():
        out = model(X, edge_index, edge_in=edge_in, in_w=in_weight,
                    edge_out=edge_out, out_w=out_weight)
    test_acc = metrics.accuracy_score(
        y[mask].cpu(), out.max(dim=1)[1].cpu()[mask])
    return test_acc


random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

date_time = datetime.now().strftime('%m-%d-%H:%M:%S')
log_path = os.path.join(args.log_root, args.log_path, args.save_name, date_time)

if os.path.isdir(log_path) == False:
    os.makedirs(log_path)
dataset_name = args.dataset.split('/')
if dataset_name[0] != 'telegram':
    try:
        data = pk.load(open(f'./data/fake/{args.dataset}.pk','rb'))
    except FileNotFoundError:
        data = pk.load(open(f'./data/fake_for_quaternion_new/{args.dataset}.pk','rb'))
    data = node_class_split(data, train_size_per_class=0.6, val_size_per_class=0.2)
else:
    data = load_directed_real_data(dataset=dataset_name[0], name=dataset_name[0])#.to(device)
    subset = args.dataset

size = data.y.size(-1)
data.y = data.y.long()

num_classes = (data.y.max() - data.y.min() + 1).cpu().numpy()

if data.x is None:
    data.x = in_out_degree(data.edge_index, size, data.edge_weight)
if data.edge_weight is not None:
    data.edge_weight = torch.FloatTensor(data.edge_weight)
data = data.to(device)

model = DiGCL(in_channels=data.x.shape[1], activation=args.activation,
              num_hidden=args.num_filter, num_proj_hidden=64,
              tau=args.tau, num_layers=2).to(device)
criterion = torch.nn.NLLLoss()

splits = data.train_mask.shape[1]

results = np.zeros((splits, 2))
alpha_1 = 0.1
for split in range(splits):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    edge_index = data.edge_index
    edge_weight = data.edge_weight
    X = data.x
    edge_index_init, edge_weight_init = cal_fast_appr(
        alpha_1, edge_index, X.shape[0], X.dtype, edge_weight=edge_weight)

    num_epochs = args.epochs
    best_test_err = 1000.0
    early_stopping = 0
    for epoch in range(args.epochs):
        a = 0.9
        b = 0.1

        if args.curr_type == 'linear':
            alpha_2 = a-(a-b)/(num_epochs+1)*epoch
        elif args.curr_type == 'exp':
            alpha_2 = a - (a-b)/(np.exp(3)-1) * \
                (np.exp(3*epoch/(num_epochs+1))-1)
        elif args.curr_type == 'log':
            alpha_2 = a - (a-b)*(1/3*np.log(epoch/(num_epochs+1)+np.exp(-3)))
        elif args.curr_type == 'fixed':
            alpha_2 = 0.9
        else:
            print('wrong curr type')
            exit()
        #loss = 0.0
        loss = train(X, edge_index,
                     alpha_1, alpha_2,
                     args.drop_feature_rate_1, args.drop_feature_rate_2, edge_weight=edge_weight)
        #print(
        #    f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {loss:.4f}')

            ####################
            # Save weights
            ####################
        save_perform = loss#.detach().item()
        if save_perform <= best_test_err:
            early_stopping = 0
            best_test_err = save_perform
            torch.save(model.state_dict(), log_path + '/model'+str(split)+'.t7')
        else:
            early_stopping += 1
        if early_stopping > 500 or epoch == (args.epochs-1):
            torch.save(model.state_dict(), log_path + '/model_latest'+str(split)+'.t7')
            break


    model.load_state_dict(torch.load(log_path + '/model'+str(split)+'.t7'))
    model.eval()
    z = model(X, edge_index_init, edge_weight_init)
    pred = pred_digcl_node(z, y=data.y,
                           train_index=data.train_mask[:, split].cpu(),
                           test_index=data.test_mask[:, split].cpu())
    pred = torch.Tensor(pred)
    #pred_label = pred.max(dim = 1)[1]

    test_acc = acc(pred, data.y, data.test_mask[:,split])


    #test_acc = metrics.accuracy_score(data.y[data.test_mask[:,split]].cpu(), pred)

    print(f'Split: {split:02d}, Test_Acc: {test_acc:.4f}')
    
    results[split] = [split, test_acc]
    torch.cuda.empty_cache() 
    model.reset_parameters()


dir_name = os.path.join(os.path.dirname(os.path.realpath(
            __file__)), '../result_arrays',args.log_path,args.dataset+'/')
if os.path.isdir(dir_name) == False:
    try:
        os.makedirs(dir_name)
    except FileExistsError:
        print('Folder exists!')            
#save_name = args.method_name + 'lr' + str(int(args.lr*1000)) + 'num_filters' + str(int(args.num_filter)) + '_curr-type_' + str(args.curr_type) 


np.save(dir_name+save_name, results)



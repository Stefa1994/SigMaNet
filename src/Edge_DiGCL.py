
import os.path as osp
import argparse
import os, time
from datetime import datetime


import torch
from sklearn import metrics
import numpy as np
import networkx as nx
import pickle as pk
import random
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')


from torch_geometric_signed_directed.data import load_directed_real_data
from utils.edge_data import link_class_split, in_out_degree,  get_appr_directed_adj, get_second_directed_adj, load_signed_real_data_no_negative

from torch_geometric_signed_directed.utils import (
    cal_fast_appr, drop_feature, pred_digcl_link)
from torch_geometric_signed_directed.nn.directed import DiGCL
from utils.edge_data import in_out_degree
from utils.edge_data_new import link_class_split_new


# select cuda device if available
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--log_root', type=str, default='../logs/', help='the path saving model.t7 and the training process')
parser.add_argument('--log_path', type=str, default='test', help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/', help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')
parser.add_argument('--dataset', type=str, default='WebKB/Cornell', help='data set selection')

parser.add_argument('--split_prob', type=lambda s: [float(item) for item in s.split(',')], default="0.05,0.15", help='random drop for testing/validation/training edges (for 3-class classification only)')
parser.add_argument('--task', type=str, default='direction', help='Task')

parser.add_argument('--activation', type=str, default = 'relu')
parser.add_argument('--tau', type=float, default =0.4)


parser.add_argument('--num_class_link', type=int, default=2,
                        help='number of classes for link direction prediction(2 or 3).')

parser.add_argument('--method_name', type=str, default='DiGCL', help='method name')

parser.add_argument('--num_filter', type=int, default=2, help='num of filters')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--drop_feature_rate_1', type=float, default=0.3)
parser.add_argument('--drop_feature_rate_2', type=float, default=0.4)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--noisy',  action='store_true')

parser.add_argument('--curr-type', type=str, default='log')

args = parser.parse_args()


save_name = args.method_name + 'lr' + str(int(args.lr*1000)) + 'num_filters' + str(int(args.num_filter)) + '_curr-type_' + str(args.curr_type) + '_activation_' +str(args.activation) + '_tau_' + str(float(args.tau)) + '_task_' + str(args.task) 
args.save_name = save_name


def acc(pred, label):
    correct = pred.eq(label).sum().item()
    acc = correct / len(pred)
    return acc

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
if len(dataset_name) == 1:
    if args.dataset in ['bitcoin_alpha', 'bitcoin_otc']:
        data = load_signed_real_data_no_negative(dataset=args.dataset).to(device)
    else:
        try:
            data = pk.load(open(f'./data/fake/{args.dataset}.pk','rb'))
        except:
            data = pk.load(open(f'./data/fake_for_quaternion_new/{args.dataset}.pk','rb'))
        data = data.to(device)
    subset = args.dataset
else:
    load_func, subset = args.dataset.split('/')[0], args.dataset.split('/')[1]
 #save_name = args.method_name + '_' + 'Layer' + str(args.layer) + '_' + 'lr' + str(args.lr) + 'num_filters' + str(int(args.num_filter))+ '_' + 'task' + str((args.task))
    data = load_directed_real_data(dataset=dataset_name[0], name=dataset_name[1]).to(device)

edge_index = data.edge_index

size = torch.max(edge_index).item()+1
data.num_nodes = size


#if data.x is None:
#    data.x = in_out_degree(data.edge_index, size, data.edge_weight)
#if data.edge_weight is not None:
#    data.edge_weight = torch.FloatTensor(data.edge_weight)

save_file = args.data_path + args.dataset + '/' + subset
#datasets = link_class_split(data, prob_val=args.split_prob[0], prob_test=args.split_prob[1], splits = 10, task = args.task, noisy = args.noisy)
datasets = link_class_split_new(data, prob_val=args.split_prob[0], prob_test=args.split_prob[1], splits = 10, task = args.task)


criterion = torch.nn.NLLLoss()

#splits = data.train_mask.shape[1]

results = np.zeros((10, 2))
alpha_1 = 0.1
for split in range(10):
    edge_index = datasets[split]['graph']
    edge_weight = datasets[split]['weights']
    edge_weight = torch.FloatTensor(edge_weight)



    X = in_out_degree(edge_index, size, edge_weight).to(device)
    edge_weight = edge_weight.to(device)
    edge_index = edge_index.long().to(device)

    model = DiGCL(in_channels=X.size(-1), activation=args.activation,
              num_hidden=args.num_filter, num_proj_hidden=64,
              tau=args.tau, num_layers=2).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    edge_index_init, edge_weight_init = cal_fast_appr(
        alpha_1, edge_index, X.shape[0], X.dtype, edge_weight=edge_weight)

    num_epochs = args.epochs
    best_test_err = 1000.0
    early_stopping = 0
    for epoch in range(num_epochs):
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
        loss = 0.0
        loss = train(X, edge_index,
                     alpha_1, alpha_2,
                     args.drop_feature_rate_1, args.drop_feature_rate_2, edge_weight=edge_weight)
        #print(
        #    f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {loss:.4f}')

                   ####################
            # Save weights
            ####################
        save_perform = loss
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
    query_train = datasets[split]['train']['edges'].cpu()
    query_test = datasets[split]['test']['edges'].cpu()
    y = datasets[split]['train']['label'].cpu()
    test_y = datasets[split]['test']['label'].cpu()
    pred = pred_digcl_link(
        z, y=y, train_index=query_train, test_index=query_test)
    pred = torch.Tensor(pred)
    #pred_label = pred.max(dim = 1)[1]

    
    test_acc = acc(pred, test_y)


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




np.save(dir_name+save_name, results)



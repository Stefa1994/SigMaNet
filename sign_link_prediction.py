import os
import sys
import time

from sklearn import metrics
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from typing import Tuple, Union

import numpy as np
#from tensorboardX import SummaryWriter
from torch_geometric_signed_directed.utils import link_class_split, in_out_degree
from torch_geometric_signed_directed.data import load_signed_real_data, SignedData
from torch_geometric_signed_directed.nn.signed import SGCN, SDGNN, SiGAT, SNEA
#from torch_geometric_signed_directed.utils.signed import link_sign_prediction_logistic_function
from src.utils.edge_data_new import link_class_split_new
import random
from scipy.sparse import coo_matrix
import torch.nn.functional as F
from src.utils.edge_data import read_edge_list_2


from src.layer.MSGNN import MSGNN_link_prediction
from src.layer.Signum_quaternion import QuaNet_link_prediction_one_laplacian
from src.layer.src2 import quaternion_laplacian
from src.layer.SSSNET_link_prediction import SSSNET_link_prediction
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(
        __file__)), 'SigMaNet'))
from src.layer.Signum import SigMaNet_link_prediction_one_laplacian
from src.layer.src2 import laplacian
import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bitcoin_otc')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--year', type=int, default=2000)

    parser.add_argument('--split_prob', type=lambda s: [float(item) for item in s.split(',')], default="0.00,0.20", help='random drop for testing/validation/training edges (for 3-class classification only)')
    parser.add_argument('--task', type=str, default='four_class_signed_digraph', help='Task')

    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes.')
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--q', type=float, default=11, help='55 means 0.5/max_{i,j}(A_{i,j} - A_{j,i}), 11, 22, 33 and 44 takes 1/5, 1/5, 3/5, 4/5 of this amount.')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--normalization', type=str, default='sym')
    parser.add_argument('--trainable_q', action='store_true')

    parser.add_argument('--emb_loss_coeff', type=float, default=0, help='Coefficient for the embedding loss term.')
    parser.add_argument('--method', type=str, default='QuaterGCN')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--in_dim', type=int, default=20)
    parser.add_argument('--out_dim', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=2)

    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')

    parser.add_argument('--debug','-D', action='store_true',
                            help='debug mode')
    parser.add_argument('--hop', type=int, default=2,
                        help='Number of hops to consider for the random walk.') 
    parser.add_argument('--tau', type=float, default=0.5,
                        help='the regularization parameter when adding self-loops to the positive part of adjacency matrix, i.e. A -> A + tau * I, where I is the identity matrix.')
    return parser.parse_args()

# torch.autograd.detect_anomaly()

args = parameter_parser()

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


from sklearn import linear_model, metrics


def link_sign_direction_prediction_logistic_function(embeddings: np.ndarray, train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, test_y: np.ndarray, class_weight: Union[dict, str] = None) -> Tuple[float, float, float, float, float]:
    """
    link_sign_prediction_logistic_function [summary]
    Link sign prediction is a binary classification machine learning task. 
    It will return the metrics for link sign prediction (i.e., Accuracy, Binary-F1, Macro-F1, Micro-F1 and AUC).
    Args:
        embeddings (np.ndarray): The embeddings for signed graph.
        train_X (np.ndarray): The indices for training data (e.g., [[0, 1], [0, 2]])
        train_y (np.ndarray): The sign for training data (e.g., [[1, -1]])
        test_X (np.ndarray): The indices for test data (e.g., [[1, 2]])
        test_y (np.ndarray): The sign for test data (e.g., [[1]])
        class_weight (Union[dict, str], optional): Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.. Defaults to None.
    Returns:
        [type]: The metrics for link sign prediction task.
        Tuple[float,float,float,float]: Accuracy, Binary-F1, Macro-F1, Micro-F1
    """
    train_X1 = []
    test_X1 = []

    for i, j in train_X:
        train_X1.append(np.concatenate([embeddings[i], embeddings[j]]))

    for i, j in test_X:
        test_X1.append(np.concatenate([embeddings[i], embeddings[j]]))

    logistic_function = linear_model.LogisticRegression(
        solver='lbfgs', max_iter=1000)
    logistic_function.fit(train_X1, train_y)
    pred = logistic_function.predict(test_X1)
    accuracy = metrics.accuracy_score(test_y, pred)
    f1        =  metrics.f1_score(test_y, pred, average='weighted')
    f1_macro = metrics.f1_score(test_y, pred, average='macro')
    f1_micro = metrics.f1_score(test_y, pred, average='micro')
    return accuracy, f1, f1_macro, f1_micro


def in_out_degree_2(edge_index:torch.LongTensor, size:int, weight=None) -> torch.Tensor:
    r"""
    Get the in degrees and out degrees of nodes
    Arg types:
        * **edge_index** (torch.LongTensor) The edge index from a torch geometric data / DirectedData object . 
        * **size** (int) - The node number.
    Return types:
        * **degree** (Torch.Tensor) - The degree matrix (|V|*2).
    """

    cpu_edge_index = edge_index.cpu()
    if weight is None:
        A = coo_matrix((np.ones(len(cpu_edge_index.T)), (cpu_edge_index[0], cpu_edge_index[1])), shape=(size, size), dtype=np.float32).tocsr()
    else:
        A = coo_matrix((weight, (cpu_edge_index[0], cpu_edge_index[1])), shape=(size, size), dtype=np.float32).tocsr()

    out_degree = np.sum(np.abs(A), axis = 0).T
    in_degree = np.sum(np.abs(A), axis = 1)
    degree = torch.from_numpy(np.c_[in_degree, out_degree]).float()
    return degree

def train_MSGNN(X_real, X_img, y, edge_index, edge_weight, query_edges):
    model.train()
    out = model(X_real, X_img, edge_index=edge_index, 
                    query_edges=query_edges, 
                    edge_weight=edge_weight)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_acc = metrics.accuracy_score(y.cpu(), out.max(dim=1)[1].cpu())
    return loss.detach().item(), train_acc

def test_MSGNN(X_real, X_img, y, edge_index, edge_weight, query_edges):
    model.eval()
    with torch.no_grad():
        out = model(X_real, X_img, edge_index=edge_index, 
                    query_edges=query_edges, 
                    edge_weight=edge_weight)
    test_y = y.cpu()
    pred = out.max(dim=1)[1].detach().cpu().numpy()
    pred_p = out.detach().cpu().numpy()
    test_acc  = metrics.accuracy_score(test_y, pred)
    f1        =  metrics.f1_score(test_y, pred,  average='weighted')
    f1_macro  =  metrics.f1_score(test_y, pred, average='macro')
    f1_micro  =  metrics.f1_score(test_y, pred, average='micro')
    #auc_score =  metrics.roc_auc_score(test_y, pred_p[:, 1])
    return test_acc, f1, f1_macro, f1_micro#, auc_score

def train_SSSNET(features, edge_index_p, edge_weight_p,
                edge_index_n, edge_weight_n, query_edges, y):
    model.train()
    out = model(edge_index_p, edge_weight_p,
            edge_index_n, edge_weight_n, features, query_edges)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_acc = metrics.accuracy_score(y.cpu(), out.max(dim=1)[1].cpu())
    return loss.detach().item(), train_acc

def test_SSSNET(features, edge_index_p, edge_weight_p,
                edge_index_n, edge_weight_n, query_edges, y):
    model.eval()
    with torch.no_grad():
        out = model(edge_index_p, edge_weight_p,
                edge_index_n, edge_weight_n, features, query_edges)
    test_y = y.cpu()
    pred = out.max(dim=1)[1].detach().cpu().numpy()
    pred_p = out.detach().cpu().numpy()
    test_acc  = metrics.accuracy_score(test_y, pred)
    f1        =  metrics.f1_score(test_y, pred, average='weighted')
    f1_macro  =  metrics.f1_score(test_y, pred, average='macro')
    f1_micro  =  metrics.f1_score(test_y, pred, average='micro')
    #auc_score =  metrics.roc_auc_score(test_y, pred_p[:, 1])
    return test_acc, f1, f1_macro, f1_micro#, auc_score

def test(train_X, test_X, train_y, test_y):
    model.eval()
    with torch.no_grad():
        z = model()

    embeddings = z.cpu().numpy()
    accuracy, f1, f1_macro, f1_micro = link_sign_direction_prediction_logistic_function(embeddings, train_X, train_y, test_X, test_y)
    return f1, f1_macro, f1_micro, accuracy

def train():
    model.train()
    optimizer.zero_grad()
    loss = model.loss()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_QuaterGCN(X_real, X_img_i, X_img_j, X_img_k, y, query_edges):
    model.train()
    out = model(X_real, X_img_i, X_img_j, X_img_k, query_edges)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_acc = metrics.accuracy_score(y.cpu(), out.max(dim=1)[1].cpu())
    return loss.detach().item(), train_acc

def test_QuaterGCN(X_real, X_img_i, X_img_j, X_img_k, y, query_edges):
    model.eval()
    with torch.no_grad():
        out = model(X_real, X_img_i, X_img_j, X_img_k, query_edges)
    test_y = y.cpu()
    pred = out.max(dim=1)[1].detach().cpu().numpy()
    pred_p = out.detach().cpu().numpy()
    test_acc  = metrics.accuracy_score(test_y, pred)
    f1        =  metrics.f1_score(test_y, pred,  average='weighted')
    f1_macro  =  metrics.f1_score(test_y, pred, average='macro')
    f1_micro  =  metrics.f1_score(test_y, pred, average='micro')
    #auc_score =  metrics.roc_auc_score(test_y, pred_p[:, 1])
    return test_acc, f1, f1_macro, f1_micro#, auc_score


def train_SigMaNet(X_real, X_img, y, query_edges):
    model.train()
    out = model(X_real, X_img, query_edges)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_acc = metrics.accuracy_score(y.cpu(), out.max(dim=1)[1].cpu())
    return loss.detach().item(), train_acc

def test_SigMaNet(X_real, X_img, y, query_edges):
    model.eval()
    with torch.no_grad():
        out = model(X_real, X_img, query_edges)
    test_y = y.cpu()
    pred = out.max(dim=1)[1].detach().cpu().numpy()
    pred_p = out.detach().cpu().numpy()
    test_acc  = metrics.accuracy_score(test_y, pred)
    f1        =  metrics.f1_score(test_y, pred,  average='weighted')
    f1_macro  =  metrics.f1_score(test_y, pred, average='macro')
    f1_micro  =  metrics.f1_score(test_y, pred, average='micro')
    #auc_score =  metrics.roc_auc_score(test_y, pred_p[:, 1])
    return test_acc, f1, f1_macro, f1_micro #, auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


assert args.method in ['SSSNET', 'SigMaNet', 'MSGNN', 'QuaterGCN', 'SGCN', 'SDGNN', 'SiGAT', 'SNEA'], 'Method not implemented'
# Download Dataset

if args.dataset in ['bitcoin_alpha', 'bitcoin_otc', 'slashdot', 'epinions']:
    data = load_signed_real_data(dataset=args.dataset).to(device)
else:
    data = read_edge_list_2(path = f"./data/wikirfa/edges.csv")

sub_dir_name = 'runs' + str(args.runs) + 'epochs' + str(args.epochs) + \
       '1000lr' + str(int(1000*args.lr)) + '1000weight_decay' + \
        str(int(1000*args.weight_decay)) + '100dropout' + str(int(100*args.dropout)) + 'task' + str(args.task)

if args.method == 'QuaterGCN':
    suffix = args.method
elif args.method == 'MSGNN':
    suffix = args.method + 'K' + str(args.K) + 'q' + str(int(100*args.q)) + 'hidden' + str(args.hidden)
elif args.method == 'SigMaNet':
    suffix = args.method
elif args.method == 'SSSNET':
    suffix =  args.method + 'hidden' + str(args.hidden) + 'hop' + str(args.hop) + '100tau' + str(int(100*args.tau))
else:
    suffix = args.method + 'in_dim' + str(args.in_dim) + 'out_dim' + str(args.out_dim)

if args.method in ['SSSNET', 'SigMaNet', 'MSGNN', 'QuaterGCN']:
    num_input_feat = 2


logs_folder_name = 'runs'

log_dir = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), args.dataset, args.method, sub_dir_name)


save_data_path_dir = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../result_arrays', args.dataset)
save_data_path = os.path.join(save_data_path_dir, 'link_sign' + str(device) + 'seed' + str(args.seed) + 'split' + str(args.runs) + '.pt')


link_data =  link_class_split_new(data, prob_val=args.split_prob[0], prob_test=args.split_prob[1], splits = args.runs, task = args.task)

nodes_num = data.num_nodes
in_dim = args.in_dim
out_dim = args.out_dim

criterion = nn.NLLLoss()
start = time.time()
res_array = np.zeros((args.runs, 4))
for split in list(link_data.keys()):
    edge_index = link_data[split]['graph']
    edge_weight = link_data[split]['weights']

    edge_i_list = edge_index.t().cpu().numpy().tolist()
    edge_weight_s = torch.where(edge_weight > 0, 1, -1)
    edge_s_list = edge_weight_s.long().cpu().numpy().tolist()
    edge_index_s = torch.LongTensor([[i, j, s] for (i, j), s in zip(edge_i_list, edge_s_list)]).to(device)
    query_edges = link_data[split]['train']['edges']
    y = link_data[split]['train']['label'].to(device)
    X_real = in_out_degree_2(edge_index, nodes_num, edge_weight).to(device)
    X_img = X_real.clone()

    if args.method == 'SGCN':
        model = SGCN(nodes_num, edge_index_s, in_dim, out_dim, layer_num=2, lamb=5).to(device)
    elif args.method == 'SNEA':
        model = SNEA(nodes_num, edge_index_s, in_dim, out_dim, layer_num=2, lamb=5).to(device)
    elif args.method == 'SiGAT':
        model = SiGAT(nodes_num, edge_index_s, in_dim, out_dim).to(device)
    elif args.method == 'SDGNN':
        model = SDGNN(nodes_num, edge_index_s, in_dim, out_dim).to(device)
    elif args.method == 'MSGNN':
        model = MSGNN_link_prediction(q=args.q, K=args.K, num_features=num_input_feat, hidden=args.hidden, label_dim=args.num_classes, \
            trainable_q = False, layer=args.num_layers, dropout=args.dropout, normalization=args.normalization, cached=(not args.trainable_q)).to(device)
    elif args.method == 'SSSNET':
        model = SSSNET_link_prediction(nfeat=num_input_feat, hidden=args.hidden, nclass=args.num_classes, dropout=args.dropout, 
        hop=args.hop, fill_value=args.tau, directed=data.is_directed).to(device)
        data1 = SignedData(edge_index=edge_index, edge_weight=edge_weight).to(device)
        data1.separate_positive_negative()
    elif args.method == 'SigMaNet':
        edge_index, norm_real, norm_imag = laplacian.process_magnetic_laplacian(edge_index=edge_index, gcn=False, net_flow=True, x_real=X_real, edge_weight=edge_weight, \
         normalization = 'sym', return_lambda_max = False)
        model = SigMaNet_link_prediction_one_laplacian(K=1, num_features=num_input_feat, hidden=args.hidden, label_dim=args.num_classes,
                            i_complex = False,  layer=args.num_layers, follow_math=False, gcn =False, net_flow=True, unwind = True, edge_index=edge_index,\
                            norm_real=norm_real, norm_imag=norm_imag,  dropout=args.dropout).to(device)
    elif args.method == 'QuaterGCN':
        edge_index, norm_real, norm_imag_i, norm_imag_j, norm_imag_k  = quaternion_laplacian.process_quaternion_laplacian(edge_index=edge_index, x_real=X_real, edge_weight=edge_weight, \
         normalization = 'sym', return_lambda_max = False)
        model = QuaNet_link_prediction_one_laplacian(K=args.K, num_features=num_input_feat, hidden=args.hidden, label_dim=args.num_classes,
                            layer=args.num_layers, unwind = True, edge_index=edge_index,\
                            norm_real=norm_real, norm_imag_i=norm_imag_i, norm_imag_j=norm_imag_j, norm_imag_k=norm_imag_k, \
                            quaternion_weights=True, quaternion_bias=True,  dropout=args.dropout).to(device)
        X_img_i = X_real.clone()
        X_img_j = X_real.clone()
        X_img_k = X_real.clone()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    query_test_edges = link_data[split]['test']['edges']
    y_test = link_data[split]['test']['label']  
    if args.method == 'MSGNN':
        for epoch in range(args.epochs):
            train_loss, train_acc = train_MSGNN(X_real, X_img, y, edge_index, edge_weight, query_edges)
            #print(f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {train_loss:.4f}, Train_Acc: {train_acc:.4f}')
            #best_run(train_loss, best_traion_err, log_path, early_stopping):

            #writer.add_scalar('train_loss_'+str(split), train_loss, epoch)

        accuracy, f1, f1_macro, f1_micro = test_MSGNN(X_real, X_img, y_test, edge_index, edge_weight, query_test_edges)
        #print(f'Split: {split:02d}, Test_Acc: {test_acc:.4f}, F1: {f1:.4f}, F1 macro: {f1_macro:.4f}, \
        #    F1 micro: {f1_micro:.4f}, AUC: {auc:.4f}')
    elif args.method == 'SSSNET':
        for epoch in range(args.epochs):
            train_loss, train_acc = train_SSSNET(X_real, data1.edge_index_p, data1.edge_weight_p,
                                        data1.edge_index_n, data1.edge_weight_n, query_edges, y)
            #print(f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {train_loss:.4f}, Train_Acc: {train_acc:.4f}')
            #writer.add_scalar('train_loss_'+str(split), train_loss, epoch)

        accuracy, f1, f1_macro, f1_micro = test_SSSNET(X_real, data1.edge_index_p, data1.edge_weight_p,
                                        data1.edge_index_n, data1.edge_weight_n, query_test_edges, y_test)
        #print(f'Split: {split:02d}, Test_Acc: {test_acc:.4f}, F1: {f1:.4f}, F1 macro: {f1_macro:.4f}, \
        #    F1 micro: {f1_micro:.4f}, AUC: {auc:.4f}')
    elif args.method == 'SigMaNet':
        for epoch in range(args.epochs):
            train_loss, train_acc = train_SigMaNet(X_real, X_img, y, query_edges)
            #print(f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {train_loss:.4f}, Train_Acc: {train_acc:.4f}')
            #writer.add_scalar('train_loss_'+str(split), train_loss, epoch)

        accuracy, f1, f1_macro, f1_micro = test_SigMaNet(X_real, X_img, y_test, query_test_edges)
        #print(f'Split: {split:02d}, Test_Acc: {test_acc:.4f}, F1: {f1:.4f}, F1 macro: {f1_macro:.4f}, \
        #    F1 micro: {f1_micro:.4f}, AUC: {auc:.4f}')
    elif args.method == 'QuaterGCN':
        for epoch in range(args.epochs):
            train_loss, train_acc = train_QuaterGCN(X_real, X_img_i, X_img_j, X_img_k, y, query_edges)
            #print(f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {train_loss:.4f}, Train_Acc: {train_acc:.4f}')
            #writer.add_scalar('train_loss_'+str(split), train_loss, epoch)

        accuracy, f1, f1_macro, f1_micro = test_QuaterGCN(X_real, X_img_i, X_img_j, X_img_k, y_test, query_test_edges)
        #print(f'Split: {split:02d}, Test_Acc: {test_acc:.4f}, F1: {f1:.4f}, F1 macro: {f1_macro:.4f}, \
        #    F1 micro: {f1_micro:.4f}, AUC: {auc:.4f}')
    else:
        for epoch in range(args.epochs):
            loss = train()
            #print(f'Split: {split:02d}, Epoch: {epoch:03d}, Loss: {loss:.4f}.')
            #writer.add_scalar('train_loss_'+str(split), loss, epoch)
        f1,  f1_macro, f1_micro, accuracy = test(query_edges.cpu(), query_test_edges.cpu(), y.cpu(), y_test.cpu())
        #print(f'Split: {split:02d}, '
        #    f'AUC: {auc:.4f}, F1: {f1:.4f}, MacroF1: {f1_macro:.4f}, MicroF1: {f1_micro:.4f}')
    res_array[split] = [accuracy, f1, f1_macro, f1_micro]
end = time.time()
memory_usage = torch.cuda.max_memory_allocated(device)*1e-6
print("Average Accuracy, F1, MacroF1 and MicroF1: {}".format(res_array.mean(0)))
print("{}'s total training and testing time: {}s, memory usage: {}M.".format(args.method, end-start, memory_usage))

dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), './result_arrays_sign/'+ args.dataset)


if os.path.isdir(os.path.join(dir_name, sub_dir_name, args.method)) == False:
    try:
        os.makedirs(os.path.join(dir_name, sub_dir_name, args.method))
        os.makedirs(os.path.join(dir_name, sub_dir_name, args.method, 'memory'))
    except FileExistsError:
        print('Folder exists for {}!'.format(sub_dir_name, args.method))
#print(os.path.join(dir_name, sub_dir_name, args.method))
np.save(os.path.join(dir_name, sub_dir_name, args.method, suffix), res_array)
np.save(os.path.join(dir_name, sub_dir_name, args.method, 'memory/runtime_memory_' + suffix), np.array([end-start, memory_usage]))
import numpy as np
import pandas as pd
import torch
import pickle as pk
import torch.optim as optim
from datetime import datetime
import os, time, argparse, csv
from collections import Counter
import random
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.datasets import WebKB, WikipediaNetwork, WikiCS
from torch_geometric.utils import to_undirected
from torch_geometric_signed_directed.data import load_directed_real_data
from torch_geometric_signed_directed.data.signed import load_signed_real_data, SignedDirectedGraphDataset
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
import networkx as nx
import scipy.sparse as sparse



# internal files
from layer.Signum import Signum_link_prediction_one_laplacian
from layer.src2 import laplacian
from torch_geometric_signed_directed.data import load_directed_real_data
from utils.edge_data import link_class_split, in_out_degree, link_prediction_evaluation
from utils.save_settings import write_log

# select cuda device if available
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="link prediction of SigNum")
    parser.add_argument('--log_root', type=str, default='../logs/', help='the path saving model.t7 and the training process')
    parser.add_argument('--log_path', type=str, default='test', help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/', help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')
    parser.add_argument('--dataset', type=str, default='WebKB/Cornell', help='data set selection')
    
    
    parser.add_argument('--split_prob', type=lambda s: [float(item) for item in s.split(',')], default="0.0,0.20", help='random drop for testing/validation/training edges (for 3-class classification only)')
    parser.add_argument('--task', type=str, default='direction', help='Task')
    
    parser.add_argument('--epochs', type=int, default=1500, help='training epochs')
    parser.add_argument('--num_filter', type=int, default=4, help='num of filters')
    parser.add_argument('--method_name', type=str, default='SigNum', help='method name')

    parser.add_argument('--K', type=int, default=1, help='K for cheb series')
    parser.add_argument('--layer', type=int, default=2, help='how many layers of gcn in the model, only 1 or 2 layers.')
    parser.add_argument('--netflow', '-N', action='store_true', help='if use net flow')
    parser.add_argument('--follow_math', '-F', action='store_true', help='if follow math')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout prob')
    parser.add_argument('--debug', '-D', action='store_true', help='debug mode')
    parser.add_argument('--num_class_link', type=int, default=2,
                        help='number of classes for link direction prediction(2 or 3).')

    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer')
    parser.add_argument('--noisy',  action='store_true')
    parser.add_argument('--randomseed', type=int, default=1, help='if set random seed in training')


    return parser.parse_args()

def read_edge_list_2(path):
    """
    Load edges from a txt file.
    """
    G = nx.DiGraph()
    edges =pd.read_csv(path, usecols = ['source','target', 'vote'])
    for i in range(edges.shape[0]):
        G.add_edge(int(edges.iloc[i][0]), int(edges.iloc[i][1]), weight=edges.iloc[i][2])
    A1 = nx.adjacency_matrix(G)
    s_A = sparse.csr_matrix(A1)
    coo = s_A.tocoo()
    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    data = Data(edge_index=indices, edge_weight=coo.data, num_nodes =  max(G.nodes) + 1)
    return data

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

def acc(pred, label):
    #print(pred.shape, label.shape)       
    correct = pred.eq(label).sum().item()
    acc = correct / len(pred)
    return acc

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.cofollow_mathl)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def main(args):

    random.seed(args.randomseed)
    torch.manual_seed(args.randomseed)
    np.random.seed(args.randomseed)

    date_time = datetime.now().strftime('%m-%d-%H:%M:%S')
    log_path = os.path.join(args.log_root, args.log_path, args.save_name, date_time)

    
    if os.path.isdir(log_path) == False:
        os.makedirs(log_path)
    
    
    dataset_name = args.dataset.split('/')
    data = load_signed_real_data(dataset=args.dataset).to(device)
    subset = args.dataset
    edge_index = data.edge_index
        

    size = torch.max(edge_index).item()+1
    data.num_nodes = size
    save_file = args.data_path + args.dataset + '/' + subset

    datasets = link_class_split(data, prob_val=args.split_prob[0], prob_test=args.split_prob[1], splits = 5, task = args.task, noisy = args.noisy)

    #print(datasets[0])

    results = np.zeros((5, 3, 5))
    for i in range(5):
        log_str_full = ''

        ########################################
        # get hermitian laplacian
        ########################################
        edge_index = datasets[i]['graph']
        edge_weight = datasets[i]['weights']
        X_real = in_out_degree_2(edge_index, size, edge_weight).to(device)
        #X_real = in_out_degree(edges, size).to(device)
        X_img = X_real.clone()
        #X_img = torch.zeros(X_real.size()).to( device=X_real.device)
        #exit()
        #else:
        #    X_img  = torch.ones(L_real[0].shape[-1]).unsqueeze(-1).to(device)
        #    X_real = torch.ones(L_real[0].shape[-1]).unsqueeze(-1).to(device)

        ########################################
        # initialize model and load dataset
        ########################################
        edge_index, norm_real, norm_imag = laplacian.process_magnetic_laplacian(edge_index=edge_index, gcn=False, net_flow=args.netflow, x_real=X_real, edge_weight=edge_weight, \
         normalization = 'sym', return_lambda_max = False)
        model = Signum_link_prediction_one_laplacian(K=args.K, num_features=2, hidden=args.num_filter, label_dim=args.num_class_link,
                            i_complex = False,  layer=args.layer, follow_math=args.follow_math, gcn =False, net_flow=args.netflow, unwind = True, edge_index=edge_index,\
                            norm_real=norm_real, norm_imag=norm_imag).to(device)

        #model = nn.DataParallel(model)  
        model = model.to(device)
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        y_train = datasets[i]['train']['label']
        y_test  = datasets[i]['test']['label']
        y_train = y_train.long().to(device)
        y_test  = y_test.long().to(device)

        train_index = datasets[i]['train']['edges']
        test_index = datasets[i]['test']['edges']
        
        #################################
        # Train/Validation/Test
        #################################
        best_test_err = 0.0
        best_test_acc = 0.0
        early_stopping = 0
        for epoch in range(args.epochs):
            start_time = time.time()
            if early_stopping > 500:
                break
            ####################
            # Train
            ####################
            train_loss, train_acc_full, train_acc, train_auc, train_f1_micro, train_f1_macro = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            model.train()
        
            out_1 = model(X_real, X_img, train_index)

            train_loss = F.nll_loss(out_1, y_train)
            #pred_label = out_1.max(dim = 1)[1]            
            #train_acc  = acc(pred_label, y_train)
            
            train_acc_full, train_acc, train_auc, train_f1_micro, train_f1_macro = link_prediction_evaluation(out_1, y_train)

            opt.zero_grad()
            train_loss.backward()
            opt.step()            
            ####################
            # Validation
            ####################
            train_loss, train_acc_full, train_acc, train_auc, train_f1_micro, train_f1_macro = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            model.eval()
            out_2 = model(X_real, X_img, test_index)

            #test_loss  = F.nll_loss(out_2, y_test)
            #pred_label = out_2.max(dim = 1)[1]            
            #test_acc   = acc(pred_label, y_test)
            test_acc_full, test_acc, test_auc, test_f1_micro, test_f1_macro =  link_prediction_evaluation(out_2, y_test)
            outstrtrain = 'Test f1: %.6f, test_f1_macro: %.3f' % (test_acc, test_f1_macro)
            outstrval = ' Test_f1_micro: %.6f, test_auc: %.3f' % (test_f1_micro, test_auc)            
            duration = "--- %.4f seconds ---" % (time.time() - start_time)
            log_str = ("%d / %d epoch" % (epoch, args.epochs))+outstrtrain+outstrval+duration
            #print(log_str)
            #log_str_full += log_str + '\n'
            ####################
            # Save weights
            ####################
            save_perform_macro = test_f1_macro #test_loss.detach().item()
            save_perform_auc = test_auc 
            if save_perform_macro >= best_test_err:
                early_stopping = 0
                best_test_err = save_perform_macro
                torch.save(model.state_dict(), log_path + '/model_macro'+str(i)+'.t7')
            if save_perform_auc >= best_test_acc:
                early_stopping = 0
                best_test_acc = save_perform_auc
                torch.save(model.state_dict(), log_path + '/model_auc'+str(i)+'.t7')
            else:
                early_stopping += 1
            test_acc_full, test_acc, test_auc, test_f1_micro, test_f1_macro = 0.0, 0.0, 0.0, 0.0, 0.0
        torch.save(model.state_dict(), log_path + '/model_latest'+str(i)+'.t7')
        write_log(vars(args), log_path)

        model.load_state_dict(torch.load(log_path + '/model_macro'+str(i)+'.t7'))
        model.eval()
        #out_train = model(X_real, X_img, train_index)
        out_test = model(X_real, X_img, test_index)
        test_acc_full_1, test_acc_1, test_auc_1, test_f1_micro_1, test_f1_macro_1 = link_prediction_evaluation(out_test, y_test)

        model.load_state_dict(torch.load(log_path + '/model_auc'+str(i)+'.t7'))
        model.eval()
        #out_train = model(X_real, X_img, train_index)
        out_test = model(X_real, X_img, test_index)
        test_acc_full_2, test_acc_2, test_auc_2, test_f1_micro_2, test_f1_macro_2 = link_prediction_evaluation(out_test, y_test)

        model.load_state_dict(torch.load(log_path + '/model_latest'+str(i)+'.t7'))
        model.eval()
        #out_train = model(X_real, X_img, train_index)
        out_test = model(X_real, X_img, test_index)
        test_acc_full_latest, test_acc_latest, test_auc_latest,  test_f1_micro_latest, test_f1_macro_latest = link_prediction_evaluation(out_test, y_test)
            ####################
            # Save testing results
            ####################
        log_str = ('test_acc_full_1:{test_acc_full_1:.4f}, test_acc_1: {test_acc_1:.4f}, '
                    + 'test_f1_micro_1: {test_f1_micro_1:.4f}, test_f1_macro_1: {test_f1_macro_1:.4f}')
        log_str1 = log_str.format(test_acc_full_1 = test_acc_full_1, test_acc_1 = test_acc_1, 
        test_f1_micro_1 = test_f1_micro_1, test_f1_macro_1 = test_f1_macro_1)
        log_str_full += log_str1 + '\n'
        #print(log_str)

        log_str = ('test_acc_full_2:{test_acc_full_2:.4f}, test_acc_2: {test_acc_2:.4f}, '
                    + 'test_f1_micro_2: {test_f1_micro_2:.4f}, test_f1_macro_2: {test_f1_macro_2:.4f}')
        log_str2 = log_str.format(test_acc_full_2 = test_acc_full_2, test_acc_2 = test_acc_2, 
        test_f1_micro_2 = test_f1_micro_2, test_f1_macro_2 = test_f1_macro_2)
        log_str_full += log_str2 + '\n'
        print(log_str_full)

        results[i] = [[test_acc_full_1, test_acc_1, test_auc_1, test_f1_micro_1, test_f1_macro_1],
                            [test_acc_full_2, test_acc_2, test_auc_2, test_f1_micro_2, test_f1_macro_2],
                            [test_acc_full_latest, test_acc_latest, test_auc_latest, test_f1_micro_latest, test_f1_macro_latest]]
        #with open(log_path + '/log'+str(i)+'.csv', 'w') as file:
        #    file.write(log_str_full)
        #    file.write('\n')
        torch.cuda.empty_cache()

    return results

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.epochs = 1

    save_name = args.method_name + 'lr' + str(int(args.lr*1000)) + 'num_filters' + str(int(args.num_filter)) + 'task' + args.task + 'layers' + str(args.layer) + 'net_flow' + str(args.netflow)  + '_noisy' +  str(args.noisy) + '_math' + str(args.follow_math)
    args.save_name = save_name

    args.log_path = os.path.join(args.log_path,args.method_name, args.dataset)
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
            __file__)), '../result_arrays',args.log_path,args.dataset+'/')
    if os.path.isdir(dir_name) == False:
        try:
            os.makedirs(dir_name)
        except FileExistsError:
            print('Folder exists!')

    results = main(args)
    np.save(dir_name+save_name, results)
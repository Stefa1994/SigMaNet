'''
Ho modificato io numero di splits a 3 così da non fare tutti gli esperimenti
'''


# external files
import numpy as np
import pickle as pk
import torch.optim as optim
from datetime import datetime
import os, time, argparse
import torch.nn.functional as F
from torch_geometric_signed_directed import node_class_split
from torch_geometric_signed_directed.data import load_directed_real_data
import random
import networkx as nx
import pickle as pk

# internal files
from utils.Citation import *
from layer.src2 import quaternion_laplacian
from layer.Signum_quaternion import QuaNet_node_prediction_one_laplacian
from utils.hermitian import *
from layer.sparse_magnet import *
from utils.save_settings import write_log
from utils.hermitian import hermitian_decomp_sparse
from utils.edge_data import in_out_degree
from utils.preprocess import load_syn



# select cuda device if available
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="node classification of QuaNet")
    parser.add_argument('--log_root', type=str, default='../logs/', help='the path saving model.t7 and the training process')
    parser.add_argument('--log_path', type=str, default='test', help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/', help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')
    parser.add_argument('--dataset', type=str, default='migration/migration', help='data set selection')

    parser.add_argument('--epochs', type=int, default=1, help='Number of (maximal) training epochs.')
    parser.add_argument('--method_name', type=str, default='SigNum', help='method name')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for training testing split/random graph generation.')

    parser.add_argument('--K', type=int, default=1, help='K for cheb series')
    parser.add_argument('--layer', type=int, default=2, help='How many layers of gcn in the model, default 2 layers.')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout prob')
    parser.add_argument('--qua_weights', '-W', action='store_true', help='quaternion weights option')
    parser.add_argument('--qua_bias', '-B', action='store_true', help='quaternion bias options')

    parser.add_argument('--debug', '-D', action='store_true', help='debug mode')
    parser.add_argument('--new_setting', '-NS', action='store_true', help='Whether not to load best settings.')

    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer')

    parser.add_argument('--num_filter', type=int, default=1, help='num of filters')
    parser.add_argument('--randomseed', type=int, default=0, help='if set random seed in training')
    return parser.parse_args()


def acc(pred, label, mask):
    correct = int(pred[mask].eq(label[mask]).sum().item())
    acc = correct / int(mask.sum())
    return acc

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
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
    if len(dataset_name) == 1:
        try:
            data = pk.load(open(f'./data/fake/{args.dataset}.pk','rb'))
        except:
            data = pk.load(open(f'./data/fake_for_quaternion_new/{args.dataset}.pk','rb'))
        data = node_class_split(data, train_size_per_class=0.6, val_size_per_class=0.2)
        subset = args.dataset
    else:
        load_func, subset = args.dataset.split('/')[0], args.dataset.split('/')[1]
        data = load_directed_real_data(dataset=dataset_name[0], name=dataset_name[1])#.to(device)
    dataset = data

    if not data.__contains__('edge_weight'):
        dataset.edge_weight = None
    else:
        dataset.edge_weight = torch.FloatTensor(dataset.edge_weight)

    size = dataset.y.size(-1)
    f_node, e_node = dataset.edge_index[0], dataset.edge_index[1]

    label = dataset.y.data.numpy().astype('int')
    train_mask = dataset.train_mask.data.numpy().astype('bool_')
    val_mask = dataset.val_mask.data.numpy().astype('bool_')
    test_mask = dataset.test_mask.data.numpy().astype('bool_')
    # normalize label, the minimum should be 0 as class index
    _label_ = label - np.amin(label)
    cluster_dim = np.amax(_label_)+1


    label = torch.from_numpy(_label_[np.newaxis]).to(device)
    label = label.reshape(label.size()[1])
    if dataset.x is None:
        X_real= in_out_degree(dataset.edge_index, size,  dataset.edge_weight).to(device)
        X_img_i = X_real.clone()
        X_img_j = X_real.clone()
        X_img_k = X_real.clone()
    else:
        X = dataset.x.data.numpy().astype('float32')
        X_real= torch.FloatTensor(X).to(device)
        X_img_i = torch.FloatTensor(X).to(device)
        X_img_j = torch.FloatTensor(X).to(device)
        X_img_k = torch.FloatTensor(X).to(device)

    criterion = nn.NLLLoss()
    edge_index, norm_real, norm_imag_i, norm_imag_j, norm_imag_k  = quaternion_laplacian.process_quaternion_laplacian(edge_index=dataset.edge_index, x_real=X_real, edge_weight=dataset.edge_weight, \
        normalization = 'sym', return_lambda_max = False)

    splits = train_mask.shape[1]
    if len(test_mask.shape) == 1:
        test_mask = np.repeat(test_mask[:,np.newaxis], splits, 1)

    results = np.zeros((splits, 4))
    for split in range(splits):
        log_str_full = ''

        model = QuaNet_node_prediction_one_laplacian(K=args.K, num_features=X_real.size(-1), hidden=args.num_filter, label_dim=cluster_dim,
                            layer=args.layer, unwind = True, edge_index=edge_index,\
                            norm_real=norm_real, norm_imag_i=norm_imag_i, norm_imag_j=norm_imag_j, norm_imag_k=norm_imag_k, \
                            quaternion_weights=args.qua_weights, quaternion_bias=args.qua_bias).to(device)

        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        train_index = train_mask[:,split]
        val_index = val_mask[:,split]
        test_index = test_mask[:,split]

        #################################
        # Train/Validation/Test
        #################################
        best_test_err = 1000.0
        early_stopping = 0
        for epoch in range(args.epochs):
            start_time = time.time()
            ####################
            # Train
            ####################
            count, train_loss, train_acc = 0.0, 0.0, 0.0

            # for loop for batch loading
            count += np.sum(train_index)

            model.train()
            preds = model(X_real, X_img_i, X_img_j, X_img_k)
            train_loss = criterion(preds[train_index], label[train_index])
            pred_label = preds.max(dim = 1)[1]
            #train_acc = 1.0*((pred_label[:,train_index] == label[:,train_index])).sum().detach().item()/count
            train_acc = acc(pred_label, label, train_index)
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            outstrtrain = 'Train loss:, %.6f, acc:, %.3f,' % (train_loss.detach().item(), train_acc)
            #scheduler.step()
            ####################
            # Validation
            ####################
            model.eval()
            count, test_loss, test_acc = 0.0, 0.0, 0.0

            # for loop for batch loading
            count += np.sum(val_index)
            preds = model(X_real, X_img_i, X_img_j, X_img_k)
            pred_label = preds.max(dim = 1)[1]

            test_loss = criterion(preds[val_index], label[val_index])
            #test_acc = 1.0*((pred_label[:,val_index] == label[:,val_index])).sum().detach().item()/count
            test_acc = acc(pred_label, label, val_index)
            outstrval = ' Test loss:, %.6f, acc:, %.3f,' % (test_loss.detach().item(), test_acc)

            duration = "---, %.4f, seconds ---" % (time.time() - start_time)
            log_str = ("%d ,/, %d ,epoch," % (epoch, args.epochs))+outstrtrain+outstrval+duration
            log_str_full += log_str + '\n'
            #print(log_str)

            ####################
            # Save weights
            ####################
            save_perform_err = test_loss.detach().item()
            save_perform_acc = test_acc
            if save_perform_err <= best_test_err:
                early_stopping = 0
                best_test_err = save_perform_err
                torch.save(model.state_dict(), log_path + '/model_err'+str(split)+'.t7')
            else:
                early_stopping += 1
            if early_stopping > 500 or epoch == (args.epochs-1):
                torch.save(model.state_dict(), log_path + '/model_latest'+str(split)+'.t7')
                break

        write_log(vars(args), log_path)

        ####################
        # Testing
        ####################
        model.load_state_dict(torch.load(log_path + '/model_err'+str(split)+'.t7'))
        model.eval()
        preds = model(X_real, X_img_i, X_img_j, X_img_k,)
        pred_label = preds.max(dim = 1)[1]
        np.save(log_path + '/pred_err' + str(split), pred_label.to('cpu'))
        acc_train = acc(pred_label, label, val_index)
        acc_test = acc(pred_label, label, test_index)


        model.load_state_dict(torch.load(log_path + '/model_latest'+str(split)+'.t7'))
        model.eval()
        preds = model(X_real, X_img_i, X_img_j, X_img_k,)
        pred_label = preds.max(dim = 1)[1]
        np.save(log_path + '/pred_latest' + str(split), pred_label.to('cpu'))
        acc_train_latest = acc(pred_label, label, val_index)
        acc_test_latest = acc(pred_label, label, test_index)

        ####################
        # Save testing results
        ####################
        logstr = 'val_acc: '+str(np.round(acc_train, 3))+' test_acc: '+str(np.round(acc_test,3))+' val_acc_latest: '+str(np.round(acc_train_latest,3))+' test_acc_latest: '+str(np.round(acc_test_latest,3))
        print(logstr)
        results[split] = [acc_train, acc_test, acc_train_latest, acc_test_latest]
        log_str_full += logstr
        with open(log_path + '/log'+str(split)+'.csv', 'w') as file:
            file.write(log_str_full)
            file.write('\n')
        torch.cuda.empty_cache()
    return results

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.epochs = 1
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
            __file__)), '../result_arrays',args.log_path,args.dataset+'/')
    if os.path.isdir(dir_name) == False:
        os.makedirs(dir_name)
    save_name = args.method_name + 'lr' + str(int(args.lr*1000)) + 'num_filters' + str(int(args.num_filter)) + 'layer' + str(args.layer) + '_quaternion_weights' + str(args.qua_weights)  + '_quaternion_bias' + str(args.qua_bias)
    args.save_name = save_name
    results = main(args)
    np.save(dir_name+save_name, results)

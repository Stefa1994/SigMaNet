
import os.path as osp
import argparse
import os, time
import random
import numpy as np


from sklearn.metrics import adjusted_rand_score
import scipy.sparse as sp
import torch
import pickle as pk
from datetime import datetime


from torch_geometric_signed_directed.nn import \
    SSSNET_node_clustering
from torch_geometric_signed_directed.data import \
    SignedData, SSBM
from torch_geometric_signed_directed.utils import \
    (Prob_Balanced_Normalized_Loss, Prob_Balanced_Ratio_Loss, 
     extract_network, triplet_loss_node_classification)

from torch_geometric_signed_directed.data import load_directed_real_data
from torch_geometric_signed_directed import node_class_split
from torch_geometric.utils import to_scipy_sparse_matrix
from utils.edge_data import in_out_degree




def acc(pred, label, mask):
    pred = pred.cpu()
    label = label.cpu()
    mask = mask.cpu()
    correct = int(pred[mask].eq(label[mask]).sum().item())
    acc = correct / int(mask.sum())
    return acc


def separate_positive_negative(data):
    ind = data.edge_weight > 0
    data.edge_index_p = data.edge_index[:, ind]
    data.edge_weight_p = data.edge_weight[ind]
    ind = data.edge_weight < 0
    data.edge_index_n = data.edge_index[:, ind]
    data.edge_weight_n = - data.edge_weight[ind]
    data.A_p = to_scipy_sparse_matrix(
        data.edge_index_p, data.edge_weight_p, num_nodes=data.num_nodes)
    data.A_n = to_scipy_sparse_matrix(
        data.edge_index_n, data.edge_weight_n, num_nodes=data.num_nodes)



parser = argparse.ArgumentParser()

parser.add_argument('--log_root', type=str, default='../logs/', help='the path saving model.t7 and the training process')
parser.add_argument('--log_path', type=str, default='test', help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/', help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')
parser.add_argument('--dataset', type=str, default='telegram/telegram', help='data set selection')

parser.add_argument('--num_filter', type=int, default=2, help='num of filters')
parser.add_argument('--method_name', type=str, default='SSSNET', help='method name')

parser.add_argument('--w_pbnc', type=int, default=1)
parser.add_argument('--w_pbrc', type=int, default=0)
parser.add_argument('--direction',  action='store_true')


parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--triplet_loss_ratio', type=float, default=0.1,
                    help='Ratio of triplet loss to cross entropy loss in supervised loss part. Default 0.1.')
parser.add_argument('--supervised_loss_ratio', type=float, default=50,
                    help='Ratio of factor of supervised loss part to self-supervised loss part.')
args = parser.parse_args()

save_name = args.method_name + 'lr' + str(int(args.lr*1000)) + 'num_filters' + str(int(args.num_filter)) + '_w_pbrc' + str(args.w_pbrc) +'_direction' + str(args.direction)
args.save_name = save_name
# select cuda device if available
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")


random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

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
    data = node_class_split(data, train_size_per_class=0.6, val_size_per_class=0.2, seed_size_per_class=0.1)
else:
    load_func, subset = args.dataset.split('/')[0], args.dataset.split('/')[1]
 #save_name = args.method_name + '_' + 'Layer' + str(args.layer) + '_' + 'lr' + str(args.lr) + 'num_filters' + str(int(args.num_filter))+ '_' + 'task' + str((args.task))
    data = load_directed_real_data(dataset=dataset_name[0], name=dataset_name[1])#.to(device)

#data.set_spectral_adjacency_reg_features(num_classes)
#data.node_split(train_size_per_class=0.8, val_size_per_class=0.1,
#                test_size_per_class=0.1, seed_size_per_class=0.1)


size = data.y.size(-1)
data.y = data.y.long()

num_classes = (data.y.max() - data.y.min() + 1).cpu().numpy()


if data.x is None:
    data.x = in_out_degree(data.edge_index, size, data.edge_weight)
if data.edge_weight is not None:
    data.edge_weight = torch.FloatTensor(data.edge_weight)

separate_positive_negative(data)
data = data.to(device)


print(num_classes)
loss_func_ce = torch.nn.NLLLoss()

model = SSSNET_node_clustering(nfeat=data.x.shape[1], dropout=0.5, hop=2, fill_value=0.5,
                               hidden=args.num_filter, nclass=num_classes, directed=args.direction).to(device)


def train(features, edge_index_p, edge_weight_p,
          edge_index_n, edge_weight_n, mask, seed_mask, loss_func_pbnc, y):
    model.train()
    Z, log_prob, pred, prob = model(edge_index_p, edge_weight_p,
                                 edge_index_n, edge_weight_n, features)
    loss_pbnc = loss_func_pbnc(prob[mask])
    loss_pbrc = loss_func_pbrc(prob[mask])
    try:
        loss_triplet = triplet_loss_node_classification(
        y=y[seed_mask], Z=Z[seed_mask], n_sample=features.size()[0], thre=0.1)
        loss_ce = loss_func_ce(log_prob[seed_mask], y[seed_mask])
        loss = args.supervised_loss_ratio*(loss_ce +
                                   args.triplet_loss_ratio*loss_triplet) + (args.w_pbnc * loss_pbnc + args.w_pbrc * loss_pbrc)
    
    except:
        loss =  args.w_pbnc * loss_pbnc + args.w_pbrc * loss_pbrc
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc = acc(pred, data.y, data.train_mask[:,split])
    return loss.detach().item(), train_acc


def test(features, edge_index_p, edge_weight_p,
         edge_index_n, edge_weight_n, mask, y):
    model.eval()
    with torch.no_grad():
        _, _, pred, prob = model(edge_index_p, edge_weight_p,
                              edge_index_n, edge_weight_n, features)
    
    test_acc = acc(pred, data.y, data.train_mask[:,split])
    return test_acc

splits = data.train_mask.shape[1]



results = np.zeros((splits, 2))
for split in range(splits):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_index = data.train_mask[:, split].cpu().numpy()
    val_index = data.val_mask[:, split]
    test_index = data.test_mask[:, split]
    seed_index = data.seed_mask[:, split]
    loss_func_pbnc = Prob_Balanced_Normalized_Loss(A_p=sp.csr_matrix(data.A_p)[train_index][:, train_index],
                                                   A_n=sp.csr_matrix(data.A_n)[train_index][:, train_index])
    loss_func_pbrc = Prob_Balanced_Ratio_Loss(A_p=sp.csr_matrix(data.A_p)[train_index][:, train_index],
                                                   A_n=sp.csr_matrix(data.A_n)[train_index][:, train_index])

    num_epochs = args.epochs
    best_test_err = 10000000000000000.0
    early_stopping = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train(data.x, data.edge_index_p, data.edge_weight_p,
                                      data.edge_index_n, data.edge_weight_n, train_index, seed_index, loss_func_pbnc, data.y)
        Val_acc = test(data.x, data.edge_index_p, data.edge_weight_p,
                       data.edge_index_n, data.edge_weight_n, val_index, data.y)
        #print(f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {train_loss:.4f}, Train_acc: {train_acc:.4f}, Val_acc: {Val_acc:.4f}')

        save_perform = train_loss#.detach().item()
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
    with torch.no_grad():
        _, _, pred, prob = model(data.edge_index_p, data.edge_weight_p,
                              data.edge_index_n, data.edge_weight_n, data.x)
    
    test_acc = acc(pred, data.y, data.train_mask[:,split])
    print(f'Split: {split:02d}, Test_acc: {test_acc:.4f}')
    results[split] = [split, test_acc]
    torch.cuda.empty_cache() 
    #model.reset_parameters()

dir_name = os.path.join(os.path.dirname(os.path.realpath(
            __file__)), '../result_arrays',args.log_path,args.dataset+'/')
if os.path.isdir(dir_name) == False:
    try:
        os.makedirs(dir_name)
    except FileExistsError:
        print('Folder exists!')            
#save_name = args.method_name + 'lr' + str(int(args.lr*1000)) + 'num_filters' + str(int(args.num_filter)) + '_curr-type_' + str(args.curr_type)  + '_w_pbnc_' + str(int(args.w_pbnc)) + '_w_pbrc_' + str(int(args.w_pbrc))


np.save(dir_name+save_name, results)
import torch
import pickle as pk
import networkx as nx
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
from torch import Tensor
from torch_sparse import SparseTensor, coalesce
#from stellargraph.data import EdgeSplitter
from sklearn.model_selection import train_test_split
from torch_geometric.utils import negative_sampling, dropout_adj, to_undirected
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_networkx
from networkx.algorithms.components import is_weakly_connected
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
from torch_scatter import scatter_add
import scipy
import os
from joblib import Parallel, delayed
from typing import Union, List, Tuple
import torch
import scipy
import numpy as np
from networkx.algorithms import tree
import torch_geometric
from scipy.sparse import coo_matrix


from typing import Optional, Callable, Union, List

from torch_geometric_signed_directed.data import SignedDirectedGraphDataset
#from .SignedDirectedGraphDatasetModified import SignedDirectedGraphDataset
from torch_geometric_signed_directed.data import SSSNET_real_data
from torch_geometric_signed_directed.data import SignedData

def negative_remove(data, double, constant):
    edge_index = data.edge_index
    row, col = edge_index[0], edge_index[1]
    size = int(max(torch.max(row), torch.max(col))+1)
    A = coo_matrix((data.edge_weight.cpu(), (row, col)), shape=(size, size), dtype=np.float32)
    mask = A.data<0
    A.data[mask]=0
    A.eliminate_zeros()
    if double:
        A.data = A.data*constant
    edge_index, weight = from_scipy_sparse_matrix(A)
    return edge_index, weight

def take_negative(data):
    edge_index = data.edge_index
    row, col = edge_index[0], edge_index[1]
    size = int(max(torch.max(row), torch.max(col))+1)
    A = coo_matrix((data.edge_weight.cpu(), (row, col)), shape=(size, size), dtype=np.float32)
    A1 = A.copy()
    mask = A1.data>0
    A1.data[mask]=0
    A1.eliminate_zeros()
    edge_index, weight = from_scipy_sparse_matrix(A1)
    return edge_index, weight   

def from_scipy_sparse_matrix(A):
    r"""Converts a scipy sparse matrix to edge indices and edge attributes.

    Args:
        A (scipy.sparse): A sparse matrix.
    """
    A = A.tocoo()
    row = torch.from_numpy(A.row).to(torch.long)
    col = torch.from_numpy(A.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_weight = torch.from_numpy(A.data)
    return edge_index, edge_weight



def load_signed_real_data_also_negative(dataset: str='epinions', root:str = './tmp_data/', double : Optional[Callable] = False,
                            constant: Union[int,float]=None, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None,
                            train_size: Union[int,float]=None, val_size: Union[int,float]=None, 
                            test_size: Union[int,float]=None, seed_size: Union[int,float]=None,
                            train_size_per_class: Union[int,float]=None, val_size_per_class: Union[int,float]=None,
                            test_size_per_class: Union[int,float]=None, seed_size_per_class: Union[int,float]=None, 
                            seed: List[int]=[], data_split: int=10) -> SignedData:
    """The function for real-world signed data downloading and convert to SignedData object.
    Arg types:
        * **dataset** (str, optional) - data set name (default: 'epinions').
        * **root** (str, optional) - The path to save the dataset (default: './').
        * **transform** (callable, optional) - A function/transform that takes in an \
            :obj:`torch_geometric.data.Data` object and returns a transformed \
            version. The data object will be transformed before every access. (default: :obj:`None`)
        * **pre_transform** (callable, optional) - A function/transform that takes in \
            an :obj:`torch_geometric.data.Data` object and returns a \
            transformed version. The data object will be transformed before \
            being saved to disk. (default: :obj:`None`)
        * **train_size** (int or float, optional) - The size of random splits for the training dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **val_size** (int or float, optional) - The size of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **test_size** (int or float, optional) - The size of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled. (Default: None. All nodes not selected for training/validation are used for testing)
        * **seed_size** (int or float, optional) - The size of random splits for the seed nodes within the training set. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **train_size_per_class** (int or float, optional) - The size per class of random splits for the training dataset. If the input is a float number, the ratio of nodes in each class will be sampled.  
        * **val_size_per_class** (int or float, optional) - The size per class of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **test_size_per_class** (int or float, optional) - The size per class of random splits for the testing dataset. If the input is a float number, the ratio of nodes in each class will be sampled. (Default: None. All nodes not selected for training/validation are used for testing)
        * **seed_size_per_class** (int or float, optional) - The size per class of random splits for seed nodes within the training set. If the input is a float number, the ratio of nodes in each class will be sampled.  
        * **seed** (An empty list or a list with the length of data_split, optional) - The random seed list for each data split.
        * **data_split** (int, optional) - number of splits (Default : 10)
    Return types:
        * **data** (Data) - The required data object.
    """
    if dataset.lower() in ['bitcoin_otc', 'bitcoin_alpha', 'slashdot', 'epinions']:
        data = SignedDirectedGraphDataset(root=root, dataset_name=dataset, transform=transform, pre_transform=pre_transform)[0]
    elif dataset.lower() in ['sp1500', 'rainfall', 'sampson', 'wikirfa', 'ppi'] or dataset[:8].lower() == 'fin_ynet':
        data = SSSNET_real_data(name=dataset, root=root, transform=transform, pre_transform=pre_transform)[0]
    else:
        raise NameError('Please input the correct data set name instead of {}!'.format(dataset))
    edge_neg, weight_neg = take_negative(data)
    edge, weight = negative_remove(data, double, constant)
    data.edge_index = edge
    data.edge_weight = weight
    signed_dataset = SignedData(edge_index=data.edge_index, edge_weight=data.edge_weight, init_data=data)
    if train_size is not None or train_size_per_class is not None:
        signed_dataset.node_split(train_size=train_size, val_size=val_size, 
            test_size=test_size, seed_size=seed_size, train_size_per_class=train_size_per_class,
            val_size_per_class=val_size_per_class, test_size_per_class=test_size_per_class,
            seed_size_per_class=seed_size_per_class, seed=seed, data_split=data_split)
    return signed_dataset, edge_neg, weight_neg




def load_signed_real_data_no_negative(dataset: str='epinions', root:str = './tmp_data/', double : Optional[Callable] = False,
                            constant: Union[int,float]=None, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None,
                            train_size: Union[int,float]=None, val_size: Union[int,float]=None, 
                            test_size: Union[int,float]=None, seed_size: Union[int,float]=None,
                            train_size_per_class: Union[int,float]=None, val_size_per_class: Union[int,float]=None,
                            test_size_per_class: Union[int,float]=None, seed_size_per_class: Union[int,float]=None, 
                            seed: List[int]=[], data_split: int=10) -> SignedData:
    """The function for real-world signed data downloading and convert to SignedData object.
    Arg types:
        * **dataset** (str, optional) - data set name (default: 'epinions').
        * **root** (str, optional) - The path to save the dataset (default: './').
        * **transform** (callable, optional) - A function/transform that takes in an \
            :obj:`torch_geometric.data.Data` object and returns a transformed \
            version. The data object will be transformed before every access. (default: :obj:`None`)
        * **pre_transform** (callable, optional) - A function/transform that takes in \
            an :obj:`torch_geometric.data.Data` object and returns a \
            transformed version. The data object will be transformed before \
            being saved to disk. (default: :obj:`None`)
        * **train_size** (int or float, optional) - The size of random splits for the training dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **val_size** (int or float, optional) - The size of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **test_size** (int or float, optional) - The size of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled. (Default: None. All nodes not selected for training/validation are used for testing)
        * **seed_size** (int or float, optional) - The size of random splits for the seed nodes within the training set. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **train_size_per_class** (int or float, optional) - The size per class of random splits for the training dataset. If the input is a float number, the ratio of nodes in each class will be sampled.  
        * **val_size_per_class** (int or float, optional) - The size per class of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **test_size_per_class** (int or float, optional) - The size per class of random splits for the testing dataset. If the input is a float number, the ratio of nodes in each class will be sampled. (Default: None. All nodes not selected for training/validation are used for testing)
        * **seed_size_per_class** (int or float, optional) - The size per class of random splits for seed nodes within the training set. If the input is a float number, the ratio of nodes in each class will be sampled.  
        * **seed** (An empty list or a list with the length of data_split, optional) - The random seed list for each data split.
        * **data_split** (int, optional) - number of splits (Default : 10)
    Return types:
        * **data** (Data) - The required data object.
    """
    if dataset.lower() in ['bitcoin_otc', 'bitcoin_alpha', 'slashdot', 'epinions']:
        data = SignedDirectedGraphDataset(root=root, dataset_name=dataset, transform=transform, pre_transform=pre_transform)[0]
    elif dataset.lower() in ['sp1500', 'rainfall', 'sampson', 'wikirfa', 'ppi'] or dataset[:8].lower() == 'fin_ynet':
        data = SSSNET_real_data(name=dataset, root=root, transform=transform, pre_transform=pre_transform)[0]
    else:
        raise NameError('Please input the correct data set name instead of {}!'.format(dataset))
    edge, weight = negative_remove(data, double, constant) 
    data.edge_index = edge
    data.edge_weight = weight
    signed_dataset = SignedData(edge_index=data.edge_index, edge_weight=data.edge_weight, init_data=data)
    if train_size is not None or train_size_per_class is not None:
        signed_dataset.node_split(train_size=train_size, val_size=val_size, 
            test_size=test_size, seed_size=seed_size, train_size_per_class=train_size_per_class,
            val_size_per_class=val_size_per_class, test_size_per_class=test_size_per_class,
            seed_size_per_class=seed_size_per_class, seed=seed, data_split=data_split)
    return signed_dataset


def sub_adj(edge_index, prob, seed):
    sub_train, sub_test = train_test_split(edge_index.T, test_size = prob, random_state=seed)
    sub_train, sub_val  = train_test_split(sub_train, test_size = 0.2, random_state=seed)
    return sub_train.T, sub_val.T, sub_test.T

def edges_positive(edge_index):
    # return true edges and reverse edges
    return edge_index, edge_index[[1,0]]

def edges_negative(edge_index):
    from torch_geometric.utils import to_undirected

    size = edge_index.max().item() + 1
    adj = np.zeros((size, size), dtype=np.int8)
    adj[edge_index[0], edge_index[1]] = 1
    x, y = np.where((adj - adj.T) < 0)

    reverse = torch.from_numpy(np.c_[x[:,np.newaxis],y[:,np.newaxis]])
    undirected_index = to_undirected(edge_index)
    negative = negative_sampling(undirected_index, num_neg_samples=edge_index[0].shape[0], force_undirected=False)

    _from_, _to_ = negative[0].unsqueeze(0), negative[1].unsqueeze(0)
    neg_index = torch.cat((_from_, _to_), axis = 0)
    #neg_index = torch.cat((reverse.T, neg_index), axis = 1)
    #print(edge_index.shape, reverse.shape, neg_index.shape)
    return reverse.T, neg_index

def split_negative(edge_index, prob, seed, neg_sampling = True):
    reverse, neg_index = edges_negative(edge_index)
    if neg_sampling:
        neg_index = torch.cat((reverse, neg_index), axis = 1)
    else:
        neg_index = reverse

    sub_train, sub_test = train_test_split(neg_index.T, test_size = prob, random_state=seed)
    sub_train, sub_val  = train_test_split(sub_train, test_size = 0.2, random_state=seed)
    return sub_train.T, sub_val.T, sub_test.T

def label_pairs_gen(pos, neg):
    pairs = torch.cat((pos, neg), axis=-1)
    label = np.r_[np.ones(len(pos[0])), np.zeros(len(neg[0]))]
    return pairs, label


# in-out degree calculation
def in_out_degree(edge_index, size, weight=None):
    if weight is None:
        A = coo_matrix((np.ones(len(edge_index)), (edge_index[0], edge_index[1])), shape=(size, size), dtype=np.float32).tocsr()
    else:
        A = coo_matrix((weight, (edge_index[0], edge_index[1])), shape=(size, size), dtype=np.float32).tocsr()

    out_degree = np.sum(np.abs(A), axis = 0).T
    in_degree = np.sum(np.abs(A), axis = 1)
    degree = torch.from_numpy(np.c_[in_degree, out_degree]).float()
    return degree

def undirected_label2directed_label(adj, edge_pairs, task):
    labels = np.zeros(len(edge_pairs), dtype=np.int32)
    new_edge_pairs = edge_pairs.copy()
    counter = 0
    for i, e in enumerate(edge_pairs): # directed edges
        if adj[e[0], e[1]] + adj[e[1], e[0]]  > 0: # exists an edge
            if adj[e[0], e[1]] > 0:
                if adj[e[1], e[0]] == 0: # rule out undirected edges
                    if counter%2 == 0:
                        labels[i] = 0
                        new_edge_pairs[i] = [e[0], e[1]]
                        counter += 1
                    else:
                        labels[i] = 1
                        new_edge_pairs[i] = [e[1], e[0]]
                        counter += 1
                else:
                    new_edge_pairs[i] = [e[0], e[1]]
                    labels[i] = -1
            else: # the other direction, and not an undirected edge
                if counter%2 == 0:
                    labels[i] = 0
                    new_edge_pairs[i] = [e[1], e[0]]
                    counter += 1
                else:
                    labels[i] = 1
                    new_edge_pairs[i] = [e[0], e[1]]
                    counter += 1
        else: # negative edges
            labels[i] = 2
            new_edge_pairs[i] = [e[0], e[1]]

    if task == 'existence':
        # existence prediction
        labels[labels == 2] = 1
        neg = np.where(labels == 1)[0]
        rng = np.random.default_rng(1000)
        neg_half = rng.choice(neg, size=len(neg)-np.sum(labels==0), replace=False)
        labels[neg_half] = -1

    return np.array(new_edge_pairs)[labels >= 0], labels[labels >= 0]


def noisy_undirected_label2directed_label(adj, edge_pairs, task):
    labels = np.zeros(len(edge_pairs), dtype=np.int32)
    new_edge_pairs = edge_pairs.copy()
    counter = 0
    if task != 'existence':
      for i, e in enumerate(edge_pairs): # directed edges
          if adj[e[0], e[1]] + adj[e[1], e[0]]  > 0: # exists an edge
              if adj[e[0], e[1]] > 0:
                  #if adj[e[1], e[0]] == 0: # rule out undirected edges
                  if counter%2 == 0:
                      labels[i] = 0
                      new_edge_pairs[i] = [e[0], e[1]]
                      counter += 1
                  else:
                      labels[i] = 1
                      new_edge_pairs[i] = [e[1], e[0]]
                      counter += 1
                  #else:
                  #    new_edge_pairs[i] = [e[0], e[1]]
                  #    labels[i] = 0
              else: # the other direction, and not an undirected edge
                  if counter%2 == 0:
                      labels[i] = 0
                      new_edge_pairs[i] = [e[1], e[0]]
                      counter += 1
                  else:
                      labels[i] = 1
                      new_edge_pairs[i] = [e[0], e[1]]
                      counter += 1
          else: # negative edges
              labels[i] = 2
              new_edge_pairs[i] = [e[0], e[1]]
          
    else: # in existence the undirected connection are always 0 (exist)!
        for i, e in enumerate(edge_pairs): # directed edges
          if adj[e[0], e[1]] + adj[e[1], e[0]]  > 0: # exists an edge
              if adj[e[0], e[1]] > 0:
                  if adj[e[1], e[0]] == 0: # rule out undirected edges
                    if counter%2 == 0:
                        labels[i] = 0
                        new_edge_pairs[i] = [e[0], e[1]]
                        counter += 1
                    else:
                        labels[i] = 1
                        new_edge_pairs[i] = [e[1], e[0]]
                        counter += 1
                  else:
                      new_edge_pairs[i] = [e[0], e[1]]
                      labels[i] = 0
              else: # the other direction, and not an undirected edge
                  if counter%2 == 0:
                      labels[i] = 0
                      new_edge_pairs[i] = [e[1], e[0]]
                      counter += 1
                  else:
                      labels[i] = 1
                      new_edge_pairs[i] = [e[0], e[1]]
                      counter += 1
          else: # negative edges
              labels[i] = 2
              new_edge_pairs[i] = [e[0], e[1]]



    if task == 'existence':
        # existence prediction
        labels[labels == 2] = 1
        neg = np.where(labels == 1)[0]
        rng = np.random.default_rng(1000)
        neg_half = rng.choice(neg, size=len(neg)-np.sum(labels==0), replace=False)
        labels[neg_half] = -1
    return np.array(new_edge_pairs)[labels >= 0], labels[labels >= 0]

def removeDuplicates(lst):
  return [t for t in (set(tuple(i) for i in lst))]

def link_class_split(data:torch_geometric.data.Data, size:int=None, splits:int=10, prob_test:float= 0.15, 
                     prob_val:float= 0.05, task:str= 'direction', seed:int= 0, maintain_connect:bool=True, 
                     ratio:float= 1.0, device:str= 'cpu', noisy:bool = True) -> dict:
    r"""Get train/val/test dataset for the link prediction task. 
    Arg types:
        * **data** (torch_geometric.data.Data or DirectedData object) - The input dataset.
        * **prob_val** (float, optional) - The proportion of edges selected for validation (Default: 0.05).
        * **prob_test** (float, optional) - The proportion of edges selected for testing (Default: 0.15).
        * **splits** (int, optional) - The split size (Default: 10).
        * **size** (int, optional) - The size of the input graph. If none, the graph size is the maximum index of nodes plus 1 (Default: None).
        * **task** (str, optional) - The evaluation task: all (three-class link prediction); direction (direction prediction); existence (existence prediction); sign (sign prediction). (Default: 'direction')
        * **seed** (int, optional) - The random seed for positve edge selection (Default: 0). Negative edges are selected by pytorch geometric negative_sampling.
        * **maintain_connect** (bool, optional) - If maintaining connectivity when removing edges for validation and testing. The connectivity is maintained by obtaining edges in the minimum spanning tree/forest first. These edges will not be removed for validation and testing (Default: True). 
        * **ratio** (float, optional) - The maximum ratio of edges used for dataset generation. (Default: 1.0)
        * **device** (int, optional) - The device to hold the return value (Default: 'cpu').
    Return types:
        * **datasets** - A dict include training/validation/testing splits of edges and labels. For split index i:
            * datasets[i]['graph'] (torch.LongTensor): the observed edge list after removing edges for validation and testing.
            * datasets[i]['train'/'val'/'testing']['edges'] (List): the edge list for training/validation/testing.
            * datasets[i]['train'/'val'/'testing']['label'] (List): the labels of edges:
                * If task == "existence": 0 (the directed edge exists in the graph), 1 (the edge doesn't exist). The undirected edges in the directed input graph are removed to avoid ambiguity.
                
                * If task == "direction": 0 (the directed edge exists in the graph), 1 (the edge of the reversed direction exists). The undirected edges in the directed input graph are removed to avoid ambiguity.
                
                * If task == "all": 0 (the directed edge exists in the graph), 1 (the edge of the reversed direction exists), 2 (the edge doesn't exist in both directions). The undirected edges in the directed input graph are removed to avoid ambiguity.
                
                * If task == "sign": 0 (negative edge), 1 (positive edge). This is the link sign prediction task for signed networks.
    """
    from torch_geometric.utils import to_undirected

    assert task in ["existence","direction","all","sign"], "Please select a valid task from 'existence', 'direction', 'all', and 'sign'!"
    edge_index = data.edge_index.cpu()
    row, col = edge_index[0], edge_index[1]
    if size is None:
        size = int(max(torch.max(row), torch.max(col))+1)
    if not hasattr(data, "edge_weight"):
        data.edge_weight = torch.ones(len(row))
    if data.edge_weight is None:
        data.edge_weight = torch.ones(len(row))


    if hasattr(data, "A"):
        A = data.A.tocsr()
    else:
        try:
            A = coo_matrix((data.edge_weight.cpu(), (row, col)), shape=(size, size), dtype=np.float32).tocsr()
        except:
            A = coo_matrix((data.edge_weight, (row, col)), shape=(size, size), dtype=np.float32).tocsr()
           

    len_val = int(prob_val*len(row))
    len_test = int(prob_test*len(row))
    if task not in ["existence", "direction", 'all']:
        pos_ratio = (A.toarray()>0).sum()/(A.toarray()!=0).sum()
        neg_ratio = 1 - pos_ratio
        len_val_pos = int(prob_val*len(row)*pos_ratio)
        len_val_neg = int(prob_val*len(row)*neg_ratio)
        len_test_pos = int(prob_test*len(row)*pos_ratio)
        len_test_neg = int(prob_test*len(row)*neg_ratio)

    undirect_edge_index = to_undirected(edge_index)
    neg_edges = negative_sampling(undirect_edge_index, num_neg_samples=len(
        edge_index.T), force_undirected=False).numpy().T
    neg_edges = map(tuple, neg_edges)
    neg_edges = list(neg_edges)

    undirect_edge_index = undirect_edge_index.T.tolist()
    if maintain_connect:
        assert ratio == 1, "ratio should be 1.0 if maintain_connect=True"
        G = nx.from_scipy_sparse_matrix(
            A, create_using=nx.Graph, edge_attribute='weight')
        mst = list(tree.minimum_spanning_edges(G, algorithm="kruskal", data=False))
        all_edges = list(map(tuple, undirect_edge_index))
        nmst = list(set(all_edges) - set(mst))
        if len(nmst) < (len_val+len_test):
            raise ValueError(
                "There are no enough edges to be removed for validation/testing. Please use a smaller prob_test or prob_val.")
    else:
        mst = []
        nmst = edge_index.T.tolist()

    rs = np.random.RandomState(seed)
    datasets = {}

    is_directed = not data.is_undirected()
    max_samples = int(ratio*len(edge_index.T))+1
    assert ratio <= 1.0 and ratio > 0, "ratio should be smaller than 1.0 and larger than 0"
    assert ratio > prob_val + prob_test, "ratio should be larger than prob_val + prob_test"
    for ind in range(splits):
        rs.shuffle(nmst)
        rs.shuffle(neg_edges)

        if task == 'sign':
            nmst = np.array(nmst)
            exist = np.array(np.abs(A[nmst[:, 0], nmst[:, 1]]) > 0).flatten()
            if np.sum(exist) < len(nmst):
                nmst = nmst[exist]

            pos_val_edges = nmst[np.array(A[nmst[:, 0], nmst[:, 1]] > 0).squeeze()].tolist()
            neg_val_edges = nmst[np.array(A[nmst[:, 0], nmst[:, 1]] < 0).squeeze()].tolist()

            ids_test = np.array(pos_val_edges[:len_test_pos].copy() + neg_val_edges[:len_test_neg].copy())
            ids_val = np.array(pos_val_edges[len_test_pos:len_test_pos+len_val_pos].copy() + \
                neg_val_edges[len_test_neg:len_test_neg+len_val_neg].copy())
            ids_train = np.array(pos_val_edges[len_test_pos+len_val_pos:max_samples] + \
                neg_val_edges[len_test_neg+len_val_neg:max_samples] + mst)

            labels_test = 1.0 * \
                np.array(A[ids_test[:, 0], ids_test[:, 1]] > 0).flatten()
            try:
                labels_val = 1.0 * \
                np.array(A[ids_val[:, 0], ids_val[:, 1]] > 0).flatten()
            except:
                labels_val = []
            labels_train = 1.0 * \
                np.array(A[ids_train[:, 0], ids_train[:, 1]] > 0).flatten()
            undirected_train = np.array([])
        else:
            ids_test = nmst[:len_test]+neg_edges[:len_test]
            ids_val = nmst[len_test:len_test+len_val]+neg_edges[len_test:len_test+len_val]
            if len_test+len_val < len(nmst):
                ids_train = nmst[len_test+len_val:max_samples]+neg_edges[len_test+len_val:max_samples]+mst
            else:
                ids_train = mst+neg_edges[len_test+len_val:max_samples]

          

        #ids_train = removeDuplicates(ids_train)
            if noisy:
                ids_test, labels_test = noisy_undirected_label2directed_label(A, ids_test, task)
            else:
                ids_test, labels_test = undirected_label2directed_label(A, ids_test, task)
            if noisy:
                ids_val, labels_val = noisy_undirected_label2directed_label(A, ids_val, task)
            else:
                ids_val, labels_val = undirected_label2directed_label(A, ids_val, task)
            if noisy:
                ids_train, labels_train = noisy_undirected_label2directed_label(A, ids_train, task)#, is_directed)
            else:
                ids_train, labels_train = undirected_label2directed_label(A, ids_train, task)


        # convert back to directed graph
        if task == 'direction':
            ids_train = ids_train[labels_train < 2]
            #label_train_w = label_train_w[labels_train <2]
            labels_train = labels_train[labels_train <2]

            ids_test = ids_test[labels_test < 2]
            #label_test_w = label_test_w[labels_test <2]
            labels_test = labels_test[labels_test <2]

            ids_val = ids_val[labels_val < 2]
            #label_val_w = label_val_w[labels_val <2]
            labels_val = labels_val[labels_val <2]
        # set up the observed graph and weights after splitting
        oberved_edges    = -np.ones((len(ids_train),2), dtype=np.int32)
        oberved_weight = np.zeros((len(ids_train),1), dtype=np.float32)

        

        direct = (np.abs(A[ids_train[:, 0], ids_train[:, 1]].data) > 0).flatten()
        oberved_edges[direct,0] = ids_train[direct,0]
        oberved_edges[direct,1] = ids_train[direct,1]
        oberved_weight[direct,0] = np.array(A[ids_train[direct,0], ids_train[direct,1]])#.flatten()

        direct = (np.abs(A[ids_train[:, 1], ids_train[:, 0]].data) > 0)[0]
        oberved_edges[direct,0] = ids_train[direct,1]
        oberved_edges[direct,1] = ids_train[direct,0]
        oberved_weight[direct,0] = np.array(A[ids_train[direct,1], ids_train[direct,0]])#.flatten()

        valid = (np.sum(oberved_edges, axis=-1) > 0)
        oberved_edges = oberved_edges[valid]
        oberved_weight = oberved_weight[valid]
        
        #oberved_edges = np.array(list(map(list,removeDuplicates(oberved_edges))))
        #oberved_weight = []
        #oberved_weight = [np.array(A[e[0], e[1]]) for i, e in enumerate(oberved_edges) if np.abs(A[e[0], e[1]]) > 0 ]    


        datasets[ind] = {}
        datasets[ind]['graph'] = torch.from_numpy(oberved_edges.T).long().to(device)
        datasets[ind]['weights'] = torch.from_numpy(oberved_weight.flatten()).float().to(device)
        #datasets[ind]['weights'] = torch.from_numpy(np.array(oberved_weight)).float().to(device)


        datasets[ind]['train'] = {}
        datasets[ind]['train']['edges'] = torch.from_numpy(ids_train).long().to(device)
        datasets[ind]['train']['label'] = torch.from_numpy(labels_train).long().to(device)
        #datasets[ind]['train']['weight'] = torch.from_numpy(label_train_w).float().to(device)

        datasets[ind]['val'] = {}
        datasets[ind]['val']['edges'] = torch.from_numpy(ids_val).long().to(device)
        try:
            datasets[ind]['val']['label'] = torch.from_numpy(labels_val).long().to(device)
        except:
            datasets[ind]['val']['label'] = []
        #datasets[ind]['val']['weight'] = torch.from_numpy(label_val_w).float().to(device)

        datasets[ind]['test'] = {}
        datasets[ind]['test']['edges'] = torch.from_numpy(ids_test).long().to(device)
        datasets[ind]['test']['label'] = torch.from_numpy(labels_test).long().to(device)
        #datasets[ind]['test']['weight'] = torch.from_numpy(label_test_w).float().to(device)
    return datasets

#################################################################################
# Copy from DiGCN
# https://github.com/flyingtango/DiGCN
#################################################################################
def get_appr_directed_adj(alpha, edge_index, num_nodes, dtype, edge_weight=None):
    from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
    from torch_scatter import scatter_add

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(edge_index.long(), edge_weight, fill_value, num_nodes)  
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes) 
    deg_inv = deg.pow(-1) 
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight 

    # personalized pagerank p
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense()
    p_v = torch.zeros(torch.Size([num_nodes+1,num_nodes+1]))
    p_v[0:num_nodes,0:num_nodes] = (1-alpha) * p_dense
    p_v[num_nodes,0:num_nodes] = 1.0 / num_nodes
    p_v[0:num_nodes,num_nodes] = alpha
    p_v[num_nodes,num_nodes] = 0.0
    p_ppr = p_v 

    eig_value, left_vector = scipy.linalg.eig(p_ppr.numpy(),left=True,right=False)
    eig_value = torch.from_numpy(eig_value.real)
    left_vector = torch.from_numpy(left_vector.real)
    val, ind = eig_value.sort(descending=True)

    pi = left_vector[:,ind[0]] # choose the largest eig vector
    pi = pi[0:num_nodes]
    p_ppr = p_dense
    pi = pi/pi.sum()  # norm pi

    # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi<0]) == 0

    pi_inv_sqrt = pi.pow(-0.5)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag()
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag()

    # L_appr
    L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) + torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0

    # make nan to 0
    L[torch.isnan(L)] = 0

    # transfer dense L to sparse
    L_indices = torch.nonzero(L,as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def get_second_directed_adj(edge_index, num_nodes, dtype, edge_weight=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight 
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense()
    
    L_in = torch.mm(p_dense.t(), p_dense)
    L_out = torch.mm(p_dense, p_dense.t())
    
    L_in_hat = L_in
    L_out_hat = L_out

    L_in_hat[L_out == 0] = 0
    L_out_hat[L_in == 0] = 0

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L,as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values
    
    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    pass


@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    pass


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))

def to_undirected(edge_index, edge_weight=None, num_nodes=None):
    """Converts the graph given by :attr:`edge_index` to an undirected graph,
    so that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (FloatTensor, optional): The edge weights.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    if edge_weight is not None:
        edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
    edge_index, edge_weight = coalesce(edge_index, edge_weight,  num_nodes, num_nodes)

    return edge_index, edge_weight 

def link_prediction_evaluation(out_test, y_test):
    r"""Evaluates link prediction results.
    Args:
        out_val: (torch.FloatTensor) Log probabilities of validation edge output, with 2 or 3 columns.
        out_test: (torch.FloatTensor) Log probabilities of test edge output, with 2 or 3 columns.
        y_val: (torch.LongTensor) Validation edge labels (with 2 or 3 possible values).
        y_test: (torch.LongTensor) Test edge labels (with 2 or 3 possible values).
    :rtype: 
        result_array: (np.array) Array of evaluation results, with shape (2, 5).
    """
    #out = torch.exp(out_val).detach().to('cpu').numpy()
    #y_val = y_val.detach().to('cpu').numpy()
    # possibly three-class evaluation
    #pred_label = np.argmax(out, axis = 1)
    #val_acc_full = accuracy_score(pred_label, y_val)
    # two-class evaluation
    #out = out[y_val < 2, :2]
    #y_val = y_val[y_val < 2]


    #prob = out[:,0]/(out[:,0]+out[:,1])
    #prob = np.nan_to_num(prob, nan=0.5, posinf=0)
    #val_auc = roc_auc_score(y_val, prob)
    #pred_label = np.argmax(out, axis = 1)
    #val_acc = accuracy_score(pred_label, y_val)
    #val_f1_macro = f1_score(pred_label, y_val, average='macro')
    #val_f1_micro = f1_score(pred_label, y_val, average='micro')

    out = torch.exp(out_test).detach().to('cpu').numpy()
    y_test = y_test.detach().to('cpu').numpy()
    # possibly three-class evaluation
    pred_label = np.argmax(out, axis = 1)
    test_acc_full = accuracy_score(pred_label, y_test)
    # two-class evaluation
    out = out[y_test < 2, :2]
    y_test = y_test[y_test < 2]
    

    prob = out[:,1]/(out[:,0]+out[:,1])
    prob = np.nan_to_num(prob, nan=0.5, posinf=0)
    test_auc = roc_auc_score(y_test, prob)
    pred_label = np.argmax(out, axis = 1)
    test_acc = f1_score(pred_label, y_test, average='binary')
    test_f1_macro = f1_score(pred_label, y_test, average='macro')
    test_f1_micro = f1_score(pred_label, y_test, average='micro')
    return test_acc_full, test_acc, test_auc, test_f1_micro, test_f1_macro
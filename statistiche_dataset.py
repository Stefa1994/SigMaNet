from platform import node
from typing import Optional
import torch
import argparse
import numpy as np
from torch_geometric.typing import OptTensor
import numpy as np
import networkx as nx
from torch_geometric_signed_directed.data import load_directed_real_data, load_signed_real_data
from src.utils.edge_data import load_signed_real_data_no_negative
import pickle as pk
import scipy
from src.utils.edge_data_new import link_class_split_new
from scipy.sparse import coo_matrix
from torch_geometric_signed_directed.utils import link_class_split
from torch_geometric_signed_directed.nn.signed import SGCN, SDGNN, SiGAT, SNEA
from src.utils.edge_data import read_edge_list_2
import pandas as pd
from torch_geometric_signed_directed.data import SignedData



# select cuda device if available
cuda_device = 0
device = torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="link prediction of QuaNet")
    parser.add_argument('--dataset', type=str, default='WebKB/Cornell', help='data set selection')
    return parser.parse_args()



def have_bidirectional_relationship(G, node1, node2):
    return G.has_edge(node1, node2) and G.has_edge(node2, node1)

# list of antiparalell edges
def biconnection(graph):
    row = [u  for u, v in graph.edges() if ((u != v) and (have_bidirectional_relationship(graph, u, v))) ]
    col = [v for u, v in graph.edges() if (u != v) and (have_bidirectional_relationship(graph, u, v)) ]
    return row, col

# Creation of a dictionary with initial node and weight indication
def dictionary_connection(graph):
    dictionary = {(node1,node2) : data['weight'] for node1, node2, data in graph.edges(data=True)}
    return dictionary

def biconnection_no_same_weights(graph, dictionary):
    row= [u  for u, v in graph.edges() if ((u != v) and (have_bidirectional_relationship(graph, u, v)) and (dictionary[u,v] != dictionary[v, u])) ]
    col = [v for u, v in graph.edges() if (u != v) and (have_bidirectional_relationship(graph, u, v) and dictionary[u,v] != dictionary[v, u]) ]
    return row, col

def biconnection_same_weights(graph, dictionary):
    row= [u  for u, v in graph.edges() if ((u != v) and (have_bidirectional_relationship(graph, u, v)) and (dictionary[u,v] == dictionary[v, u])) ]
    col = [v for u, v in graph.edges() if (u != v) and (have_bidirectional_relationship(graph, u, v) and dictionary[u,v] == dictionary[v, u]) ]
    return row, col

    
    
# Creation of the subdivision graph
def antiparalell_same_weights(graph):
    '''
    Extraction of the antiparalell edges with same weights
    '''
    graph_1 = nx.from_scipy_sparse_array(graph, create_using=nx.DiGraph)
    numero_edges = graph_1.edges()
    print(len(numero_edges))
    dictionary = dictionary_connection(graph_1)
    row, col = biconnection_same_weights(graph_1, dictionary)
    print('same weight', len(row))
    row, col = biconnection_no_same_weights(graph_1, dictionary)
    print('different weight', len(row))
    row, col = biconnection(graph_1)
    print(len(row))
    #return coo_matrix((np.ones(len(row)), (row, col)), shape=(graph_1.number_of_nodes(), graph_1.number_of_nodes()), dtype=np.int8)


def negative_remove(data):
    edge_index = data.edge_index
    row, col = edge_index[0], edge_index[1]
    size = int(max(torch.max(row), torch.max(col))+1)
    A = coo_matrix((data.edge_weight.cpu(), (row, col)), shape=(size, size), dtype=np.float32)
    mask = A.data<0
    A.data[mask]=0
    A.eliminate_zeros()
    edge_index, weight = from_scipy_sparse_matrix(A)
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

def main(args):
    dataset_name = args.dataset.split('/')
    if args.dataset in ['telegram']:
        data = load_directed_real_data(dataset=dataset_name[0], name=dataset_name[0]).to(device)
        data = data.to(device)
        subset = args.dataset
    else:

        #
        if args.dataset in ['bitcoin_alpha', 'bitcoin_otc', 'slashdot', 'epinions']:
            #data = load_signed_real_data(dataset=args.dataset).to(device)
            data = load_signed_real_data_no_negative(dataset=args.dataset).to(device)
        elif args.dataset in 'wikirfa':
            data = read_edge_list_2(path = f"./data/wikirfa/edges.csv").to(device)
            edge, weight = negative_remove(data) 
            data.edge_index = edge
            data.edge_weight = weight
            data = SignedData(edge_index=data.edge_index, edge_weight=data.edge_weight, init_data=data)


        else:
            try:
                data = pk.load(open(f'./data/fake/{args.dataset}.pk','rb'))
            except:
                data = pk.load(open(f'./data/fake_for_quaternion_new/{args.dataset}.pk','rb'))

    #print(data.A)
    print(data)
    edge_index = data.edge_index   
    size = torch.max(edge_index).item()+1
    data.num_nodes = size
    print(size)
    print('number of edges: ', len(data.edge_weight))
    print('density:',len(data.edge_weight)/ (size * (size - 1)) )
    print(min(data.edge_weight))
    #data = scipy.sparse.coo_matrix((data.edge_weight, (edge_index[0], edge_index[1])), shape=(size, size))  
    datasets = link_class_split(data, prob_val=0.05, prob_test=0.15, splits = 10, task = 'direction')

    
    edge_index = datasets[1]['graph']
    edge_weight = datasets[1]['weights']
    row, col = torch.split(edge_index, 1, dim=0)
    row = row.squeeze(0)
    col = col.squeeze(0)
    datasets = coo_matrix((edge_weight, (row, col)), shape=(size, size), dtype=np.float32)
    #try:
    antiparalell_same_weights(datasets)
    #except:
    #    antiparalell_same_weights(data)


if __name__ == "__main__":
    args = parse_args()
    main(args)
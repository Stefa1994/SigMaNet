import networkx as nx
from scipy.sparse import coo_matrix
import numpy as np


def have_bidirectional_relationship(G, node1, node2):
    return G.has_edge(node1, node2) and G.has_edge(node2, node1)

# list of antiparalell edges with same weights
def biconnection(graph, dictionary):
    row= [u  for u, v in graph.edges() if ((u != v) and (have_bidirectional_relationship(graph, u, v)) and (dictionary[u,v] == dictionary[v, u])) ]
    col = [v for u, v in graph.edges() if (u != v) and (have_bidirectional_relationship(graph, u, v) and dictionary[u,v] == dictionary[v, u]) ]
    return row, col

# list of antiparalell edges with different weights
def biconnection_no_same_weights(graph, dictionary):
    row= [u  for u, v in graph.edges() if ((u != v) and (have_bidirectional_relationship(graph, u, v)) and (dictionary[u,v] != dictionary[v, u])) ]
    col = [v for u, v in graph.edges() if ((u != v) and (have_bidirectional_relationship(graph, u, v) and dictionary[u,v] != dictionary[v, u])) ]
    return row, col

# Creation of a dictionary with initial node and weight indication
def dictionary_connection(graph):
    dictionary = {(node1,node2) : data['weight'] for node1, node2, data in graph.edges(data=True)}
    return dictionary

# Creation of the subdivision graph
def antiparalell(graph):
    '''
    Extraction of the antiparalell edges with same weights
    '''
    graph_1 = nx.from_scipy_sparse_matrix(graph, create_using=nx.DiGraph)
    dictionary = dictionary_connection(graph_1)
    row, col = biconnection(graph_1, dictionary)
    return coo_matrix((np.ones(len(row)), (row, col)), shape=(graph_1.number_of_nodes(), graph_1.number_of_nodes()), dtype=np.int8)

# Creation of the subdivision graph
def antiparalell_different_weights(graph):
    '''
    Extraction of the antiparalell edges with different weights
    '''
    graph_1 = nx.from_scipy_sparse_array(graph, create_using=nx.DiGraph)
    dictionary = dictionary_connection(graph_1)
    row, col = biconnection_no_same_weights(graph_1, dictionary)
    data =  [dictionary[u,v] for u, v in zip(row, col)]
    return coo_matrix((data, (row, col)), shape=(graph_1.number_of_nodes(), graph_1.number_of_nodes()), dtype=np.int8)
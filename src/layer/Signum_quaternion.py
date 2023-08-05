'''
SigMaNet architecture
'''

import torch
from torch.nn import Parameter
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.nn.conv import MessagePassing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .src2 import laplacian



class complex_relu_layer(nn.Module):
    """
    The complex ReLU layer for quaternion
    """
    def __init__(self, ):
        super(complex_relu_layer, self).__init__()
    
    def complex_relu(self, real:torch.FloatTensor, imag_i:torch.FloatTensor, imag_j:torch.FloatTensor, imag_k:torch.FloatTensor):
        """
        Complex ReLU function.
        
        Arg types:
            * real, imag_1, imag_2, imag_3 (PyTorch Float Tensor) - Node features.
        Return types:
            * real, imag_1, imag_2, imag_3 (PyTorch Float Tensor) - Node features after complex ReLU.
        """
        mask = 1.0*(real >= 0)
        return mask*real, mask*imag_i, mask*imag_j, mask*imag_k

    def forward(self, real:torch.FloatTensor, imag_i:torch.FloatTensor, imag_j:torch.FloatTensor, imag_k:torch.FloatTensor):
        """
        Making a forward pass of the complex ReLU layer.
        
        Arg types:
            * real, imag_1, imag_2, imag_3 (PyTorch Float Tensor) - Node features.
        Return types:
            * real, imag_1, imag_2, imag_3 (PyTorch Float Tensor) - Node features after complex ReLU.
        """
        real, imag_i, imag_j, imag_k = self.complex_relu(real, imag_i, imag_j, imag_k)
        return real, imag_i, imag_j, imag_k

class complex_relu_layer_different(nn.Module):
    """
    The complex ReLU layer for quaternion where a function is applied specifically to each components
    """
    def __init__(self, ):
        super(complex_relu_layer_different, self).__init__()
    
    def complex_relu(self, real:torch.FloatTensor, imag_i:torch.FloatTensor, imag_j:torch.FloatTensor, imag_k:torch.FloatTensor):
        """
        Complex ReLU function.
        
        Arg types:
            * real, imag_1, imag_2, imag_3 (PyTorch Float Tensor) - Node features.
        Return types:
            * real, imag_1, imag_2, imag_3 (PyTorch Float Tensor) - Node features after complex ReLU.
        """
        mask_r = 1.0*(real >= 0)
        mask_i = 1.0*(imag_i>= 0)
        mask_j = 1.0*(imag_j >= 0)
        mask_k = 1.0*(imag_k >= 0)
        return mask_r*real, mask_i*imag_i, mask_j*imag_j, mask_k*imag_k

    def forward(self, real:torch.FloatTensor, imag_i:torch.FloatTensor, imag_j:torch.FloatTensor, imag_k:torch.FloatTensor):
        """
        Making a forward pass of the complex ReLU layer.
        
        Arg types:
            * real, imag_1, imag_2, imag_3 (PyTorch Float Tensor) - Node features.
        Return types:
            * real, imag_1, imag_2, imag_3 (PyTorch Float Tensor) - Node features after complex ReLU.
        """
        real, imag_i, imag_j, imag_k = self.complex_relu(real, imag_i, imag_j, imag_k)
        return real, imag_i, imag_j, imag_k

class complex_Leaky_relu_layer_different(nn.Module):
    """
    The complex ReLU layer for quaternion where a function is applied specifically to each components
    """
    def __init__(self, ):
        super(complex_Leaky_relu_layer_different, self).__init__()
    
    def complex_Leaky_relu(self, real:torch.FloatTensor, imag_i:torch.FloatTensor, imag_j:torch.FloatTensor, imag_k:torch.FloatTensor):
        """
        Complex ReLU function.
        
        Arg types:
            * real, imag_1, imag_2, imag_3 (PyTorch Float Tensor) - Node features.
        Return types:
            * real, imag_1, imag_2, imag_3 (PyTorch Float Tensor) - Node features after complex ReLU.
        """
        
        leaky = torch.nn.LeakyReLU()
        
        
        return leaky(real), leaky(imag_i), leaky(imag_j), leaky(imag_k)

    def forward(self, real:torch.FloatTensor, imag_i:torch.FloatTensor, imag_j:torch.FloatTensor, imag_k:torch.FloatTensor):
        """
        Making a forward pass of the complex ReLU layer.
        
        Arg types:
            * real, imag_1, imag_2, imag_3 (PyTorch Float Tensor) - Node features.
        Return types:
            * real, imag_1, imag_2, imag_3 (PyTorch Float Tensor) - Node features after complex ReLU.
        """
        real, imag_i, imag_j, imag_k = self.complex_Leaky_relu(real, imag_i, imag_j, imag_k)
        return real, imag_i, imag_j, imag_k



class QuaNetConv(MessagePassing):    
    def __init__(self, in_channels:int, out_channels:int, K:int, normalization:str='sym', bias:bool=True, edge_index=None, 
                norm_real=None, norm_imag_i=None, norm_imag_j=None, norm_imag_k=None, quaternion_weights=False, quaternion_bias=False, **kwargs): #norm_imag_3=None, 
        kwargs.setdefault('aggr', 'add')
        super(QuaNetConv, self).__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym'], 'Invalid normalization'
        kwargs.setdefault('flow', 'target_to_source')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        #if gcn: # devo eliminare i pesi creati per moltiplicarli con il self-loop e creo solo un peso nel caso Theta moltiplica tutto [(I + A)\Theta]
        K = 1 # Because I have to stop at the first stage
        self.quaternion_weights = quaternion_weights
        self.quaternion_bias = quaternion_bias


        if self.quaternion_weights:
            self.weight = Parameter(torch.Tensor(K, 4, in_channels, out_channels)) 
        else:
            self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))


        if bias:
            if self.quaternion_bias:
                self.bias = Parameter(torch.Tensor(4, out_channels))
            else:
                self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Inserisco qui i valori di edge index, norm_real e norm_imag_1, norm_imag_2, norm_imag_3
        # la creazione i valori come self

        self.edge_index = edge_index
        self.norm_real = norm_real
        self.norm_imag_1 = norm_imag_i
        self.norm_imag_2 = norm_imag_j
        self.norm_imag_3 = norm_imag_k

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    
    def quaternion_multiplication(self, real: torch.FloatTensor, 
                                  imag_i: torch.FloatTensor, 
                                  imag_j: torch.FloatTensor,
                                  imag_k: torch.FloatTensor):
        
        # Weights definition
        weight_r = self.weight[0, 0, :]
        weight_i = self.weight[0, 1, :]
        weight_j = self.weight[0, 2, :]
        weight_k = self.weight[0, 3, :]


        # Multiplication between feature_vectors and weights 
        real_real_1 = torch.spmm(real, weight_r) - torch.spmm(imag_i, weight_i) - torch.spmm(imag_j, weight_j) - torch.spmm(imag_k, weight_k)
        imag_imag_i = torch.spmm(real, weight_i) + torch.spmm(imag_i, weight_r) + torch.spmm(imag_j, weight_k) - torch.spmm(imag_k, weight_j)
        imag_imag_j = torch.spmm(real, weight_j) - torch.spmm(imag_i, weight_k) + torch.spmm(imag_j, weight_r) + torch.spmm(imag_k, weight_i)
        imag_imag_k = torch.spmm(real, weight_k) + torch.spmm(imag_i, weight_j) - torch.spmm(imag_j, weight_i) + torch.spmm(imag_k, weight_r)

        return real_real_1, imag_imag_i, imag_imag_j, imag_imag_k


    
 
    def forward(
        self,
        X_real: torch.FloatTensor, 
        X_imag_1: torch.FloatTensor, 
        X_imag_2: torch.FloatTensor, 
        X_imag_3: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the SigMaNet Convolution layer.
        
        Arg types:
            * x_real, x_imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long TensSor) - Edge indices.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
            * lambda_max (optional, but mandatory if normalization is None) - Largest eigenvalue of Laplacian.
        Return types:
            * out_real, out_imag (PyTorch Float Tensor) - Hidden state tensor for all nodes, with shape (N_nodes, F_out).
        """
        
        self.n_dim = X_real.shape[0]
       
       # Operazione credo inutile.. ma vediamo
        norm_imag_1 = - self.norm_imag_1
        norm_imag_2 = - self.norm_imag_2
        norm_imag_3 = - self.norm_imag_3
        norm_real = - self.norm_real

        edge_index = self.edge_index

        # Propagazione dell'informazione
        # First-step
        Tx_0_real_real_1 = self.propagate(edge_index, x=X_real, norm=norm_real, size=None).to(torch.float) - self.propagate(edge_index, x=X_imag_1, norm=norm_imag_1, size=None).to(torch.float) - \
        self.propagate(edge_index, x=X_imag_2, norm=norm_imag_2, size=None).to(torch.float) - self.propagate(edge_index, x=X_imag_3, norm=norm_imag_3, size=None).to(torch.float)
        
        Tx_0_imag_imag_1 = self.propagate(edge_index, x=X_imag_1, norm=norm_real, size=None).to(torch.float) + self.propagate(edge_index, x=X_real, norm=norm_imag_1, size=None).to(torch.float) + \
        self.propagate(edge_index, x=X_imag_3, norm=norm_imag_2, size=None).to(torch.float) - self.propagate(edge_index, x=X_imag_2, norm=norm_imag_3, size=None).to(torch.float)
        
        Tx_0_imag_imag_2 = self.propagate(edge_index, x=X_imag_2, norm=norm_real, size=None).to(torch.float) - self.propagate(edge_index, x=X_imag_3, norm=norm_imag_1, size=None).to(torch.float) + \
        self.propagate(edge_index, x=X_real, norm=norm_imag_2, size=None).to(torch.float) + self.propagate(edge_index, x=X_imag_1, norm=norm_imag_3, size=None).to(torch.float)

        Tx_0_imag_imag_3 = self.propagate(edge_index, x=X_imag_3, norm=norm_real, size=None).to(torch.float) + self.propagate(edge_index, x=X_imag_2, norm=norm_imag_1, size=None).to(torch.float) - \
        self.propagate(edge_index, x=X_imag_1, norm=norm_imag_2, size=None).to(torch.float) + self.propagate(edge_index, x=X_real, norm=norm_imag_3, size=None).to(torch.float)

        # Second-step: multiplication with the weight of the neural network
        # Versione One (i pesi sono uguali per tutte le componenti)
        # In questo caso il tensore dei pesi è di 3 dimensioni --> [K , In-dimension, Out-dimension]
        # dove K = 1 perchè ci fermiamo a quel valore di K (K = 1)

        if self.quaternion_weights:
            out_real, out_imag_1, out_imag_2, out_imag_3 = self.quaternion_multiplication(Tx_0_real_real_1, Tx_0_imag_imag_1, Tx_0_imag_imag_2, Tx_0_imag_imag_3)
        else:
            out_real = torch.matmul(Tx_0_real_real_1, self.weight[0])
            out_imag_1 = torch.matmul(Tx_0_imag_imag_1, self.weight[0])
            out_imag_2 = torch.matmul(Tx_0_imag_imag_2, self.weight[0])
            out_imag_3 = torch.matmul(Tx_0_imag_imag_3, self.weight[0])


        # Si crea lo stesso scenario con i pesi normali
        if self.bias is not None:
            if self.quaternion_bias:
                out_real += self.bias[0]
                out_imag_1 += self.bias[1]
                out_imag_2 += self.bias[2]
                out_imag_3 += self.bias[3]
            else:
                out_real += self.bias
                out_imag_1 += self.bias
                out_imag_2 += self.bias
                out_imag_3 += self.bias



        return out_real, out_imag_1, out_imag_2, out_imag_3


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)

class QuaNet_link_prediction_one_laplacian(nn.Module):
    r"""The QuaNet model for link prediction from the    
    Args:
        num_features (int): Size of each input sample.
        hidden (int, optional): Number of hidden channels.  Default: 2.
        K (int, optional): Order of the Chebyshev polynomial.  Default: 2.
        label_dim (int, optional): Number of output classes.  Default: 2.
        activation (bool, optional): whether to use activation function or not. (default: :obj:`True`)
        layer (int, optional): Number of MagNetConv layers. Deafult: 2.
        dropout (float, optional): Dropout value. (default: :obj:`0.5`)
        normalization (str, optional): The normalization scheme for the magnetic
            Laplacian (default: :obj:`sym`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A} Hadamard \exp(i \Theta^{(q)})`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2} Hadamard \exp(i \Theta^{(q)})`
    """
    def __init__(self, num_features:int, hidden:int=2, K:int=2, label_dim:int=2, \
        activation:bool=True, layer:int=2, dropout:float=0.5, normalization:str='sym',\
        unwind:bool=True, edge_index=None, norm_real=None, norm_imag_i=None, norm_imag_j=None, norm_imag_k=None,\
        quaternion_weights:bool=True, quaternion_bias:bool=True):
        super(QuaNet_link_prediction_one_laplacian, self).__init__()

        chebs = nn.ModuleList()
        chebs.append(QuaNetConv(in_channels=num_features, out_channels=hidden, K=K,\
                                 normalization=normalization, edge_index=edge_index,\
                                 norm_real=norm_real, norm_imag_i=norm_imag_i, \
                                 norm_imag_j=norm_imag_j, norm_imag_k=norm_imag_k, \
                                 quaternion_weights=quaternion_weights, quaternion_bias=quaternion_bias))
        self.normalization = normalization
        self.activation = activation
        if self.activation:
            #self.complex_relu = complex_relu_layer()
            self.complex_relu = complex_relu_layer_different()
            #self.complex_relu = complex_Leaky_relu_layer_different()
        for _ in range(1, layer):
            chebs.append(QuaNetConv(in_channels=hidden, out_channels=hidden, K=K,\
                                 normalization=normalization, edge_index=edge_index,\
                                 norm_real=norm_real, norm_imag_i=norm_imag_i, \
                                 norm_imag_j=norm_imag_j, norm_imag_k=norm_imag_k, \
                                 quaternion_weights=quaternion_weights, quaternion_bias=quaternion_bias))

        self.Chebs = chebs 
        self.linear = nn.Linear(hidden*8, label_dim)   
        self.dropout = dropout
        #self.complex_dropout = drop.Dropout(self.dropout)
        #self.complex_softmax = act.modLogSoftmax(dim=1)
        self.unwind = unwind

        
    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        self.linear.reset_parameters()
        

    def forward(self, real: torch.FloatTensor, imag_1: torch.FloatTensor, imag_2: torch.FloatTensor, \
        imag_3: torch.FloatTensor, query_edges: torch.LongTensor) -> torch.FloatTensor:
        """
      
        Arg types:
            * real, imag_1, imag_2, imag_3 (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * query_edges (PyTorch Long Tensor) - Edge indices for querying labels.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * log_prob (PyTorch Float Tensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
        """
        for cheb in self.Chebs:           
            real, imag_1, imag_2, imag_3 = cheb(real, imag_1, imag_2, imag_3)
            if self.activation:
                real, imag_1, imag_2, imag_3 = self.complex_relu(real, imag_1, imag_2, imag_3)

        # Unwind operation
        x = torch.cat((real[query_edges[:,0]], real[query_edges[:,1]], imag_1[query_edges[:,0]], imag_1[query_edges[:,1]], \
        imag_2[query_edges[:,0]], imag_2[query_edges[:,1]], imag_3[query_edges[:,0]], imag_3[query_edges[:,1]]), dim = -1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x


class QuaNet_node_prediction_one_laplacian(nn.Module):
    r"""The QuaNet model for node classification 
    Args:
        num_features (int): Size of each input sample.
        hidden (int, optional): Number of hidden channels.  Default: 2.
        K (int, optional): Order of the Chebyshev polynomial.  Default: 2.
        label_dim (int, optional): Number of output classes.  Default: 2.
        activation (bool, optional): whether to use activation function or not. (default: :obj:`True`)
        layer (int, optional): Number of MagNetConv layers. Deafult: 2.
        dropout (float, optional): Dropout value. (default: :obj:`0.5`)
        normalization (str, optional): The normalization scheme for the magnetic
            Laplacian (default: :obj:`sym`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A} Hadamard \exp(i \Theta^{(q)})`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2} Hadamard \exp(i \Theta^{(q)})`
    """
    def __init__(self, num_features:int, hidden:int=2, K:int=1, label_dim:int=2, \
        activation:bool=True, layer:int=2, dropout:float=0.5, normalization:str='sym',\
        unwind:bool=False, edge_index=None, norm_real=None, norm_imag_i=None, norm_imag_j=None, norm_imag_k=None, \
        quaternion_weights:bool=False, quaternion_bias:bool=False):
        super(QuaNet_node_prediction_one_laplacian, self).__init__()

        chebs = nn.ModuleList()
        chebs.append(QuaNetConv(in_channels=num_features, out_channels=hidden, K=K,\
                                 normalization=normalization, edge_index=edge_index,\
                                 norm_real=norm_real, norm_imag_i=norm_imag_i, \
                                 norm_imag_j=norm_imag_j, norm_imag_k=norm_imag_k,\
                                 quaternion_weights=quaternion_weights, quaternion_bias=quaternion_bias))
        self.normalization = normalization
        self.activation = activation
        if self.activation:
            #self.complex_relu = complex_relu_layer()
            self.complex_relu = complex_relu_layer_different()
        for _ in range(1, layer):
            chebs.append(QuaNetConv(in_channels=hidden, out_channels=hidden, K=K,\
                                 normalization=normalization, edge_index=edge_index,\
                                 norm_real=norm_real, norm_imag_i=norm_imag_i, \
                                 norm_imag_j=norm_imag_j, norm_imag_k=norm_imag_k, \
                                 quaternion_weights=quaternion_weights, quaternion_bias=quaternion_bias))

        self.Chebs = chebs
        last_dim = 4 # era 2.. vediamo
        self.Conv = nn.Conv1d(hidden*last_dim, label_dim, kernel_size=1)
        self.dropout = dropout
        self.unwind = unwind

        
    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        self.Conv.reset_parameters()


    def forward(self, real: torch.FloatTensor, imag_1: torch.FloatTensor, imag_2: torch.FloatTensor, \
        imag_3: torch.FloatTensor) -> torch.FloatTensor:
        """
        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * log_prob (PyTorch Float Tensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
        """
        for cheb in self.Chebs:           
            real, imag_1, imag_2, imag_3 = cheb(real, imag_1, imag_2, imag_3)
            if self.activation:
                real, imag_1, imag_2, imag_3 = self.complex_relu(real, imag_1, imag_2, imag_3)

        x = torch.cat((real, imag_1, imag_2, imag_3), dim = -1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()
        x = F.log_softmax(x, dim=1)
        return x
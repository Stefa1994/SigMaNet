from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.utils import add_self_loops, remove_self_loops, to_scipy_sparse_matrix
from torch_geometric.utils.num_nodes import maybe_num_nodes
import numpy as np
from scipy.sparse.linalg import eigsh


def get_magnetic_signed_Laplacian(edge_index: torch.LongTensor, edge_weight: Optional[torch.Tensor] = None,
                  normalization: Optional[str] = 'sym',
                  dtype: Optional[int] = None,
                  num_nodes: Optional[int] = None,
                  q: Optional[float] = 0.25,
                  return_lambda_max: bool = False, 
                  absolute_degree: bool = True):
    r""" Computes the magnetic signed Laplacian of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight`.
    
    Arg types:
        * **edge_index** (PyTorch LongTensor) - The edge indices.
        * **edge_weight** (PyTorch Tensor, optional) - One-dimensional edge weights. (default: :obj:`None`)
        * **normalization** (str, optional) - The normalization scheme for the magnetic Laplacian (default: :obj:`sym`) -
            1. :obj:`None`: No normalization :math:`\mathbf{L} = \bar{\mathbf{D}} - \mathbf{A} Hadamard \exp(i \Theta^{(q)})`
            
            2. :obj:`"sym"`: Symmetric normalization :math:`\mathbf{L} = \mathbf{I} - \bar{\mathbf{D}}^{-1/2} \mathbf{A}
            \bar{\mathbf{D}}^{-1/2} Hadamard \exp(i \Theta^{(q)})`
        
        * **dtype** (torch.dtype, optional) - The desired data type of returned tensor in case :obj:`edge_weight=None`. (default: :obj:`None`)
        * **num_nodes** (int, optional) - The number of nodes, *i.e.* :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        * **q** (float, optional) - The value q in the paper for phase.
        * **return_lambda_max** (bool, optional) - Whether to return the maximum eigenvalue. (default: :obj:`False`)
        * **absolute_degree** (bool, optional) - Whether to calculate the degree matrix with respect to absolute entries of the adjacency matrix. (default: :obj:`True`)
    
    Return types:
        * **edge_index** (PyTorch LongTensor) - The edge indices of the magnetic signed Laplacian.
        * **edge_weight.real, edge_weight.imag** (PyTorch Tensor) - Real and imaginary parts of the one-dimensional edge weights for the magnetic signed Laplacian.
        * **lambda_max** (float, optional) - The maximum eigenvalue of the magnetic signed Laplacian, only returns this when required by setting return_lambda_max as True.
    """

    if normalization is not None:
        assert normalization in ['sym'], 'Invalid normalization'

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                 device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    theta_attr = torch.cat([edge_weight, -edge_weight], dim=0) # type: ignore
    sym_attr = torch.cat([edge_weight, edge_weight], dim=0)
    sym_abs_attr = torch.cat([torch.abs(edge_weight), torch.abs(edge_weight)], dim=0)
    edge_attr = torch.stack([sym_attr, theta_attr, sym_abs_attr], dim=1)

    edge_index_sym, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes, "add")

    row, col = edge_index_sym[0], edge_index_sym[1]
    edge_weight_sym = edge_attr[:, 0]
    edge_weight_sym = edge_weight_sym/2

    if absolute_degree:
        edge_weight_abs_sym = edge_attr[:, 2]
        edge_weight_abs_sym = edge_weight_abs_sym/2
        deg = scatter_add(edge_weight_abs_sym, row, dim=0, dim_size=num_nodes)
    else:
        deg = scatter_add(torch.abs(edge_weight_sym), row, dim=0, dim_size=num_nodes) # absolute values for edge weights

    edge_weight_q = torch.exp(1j * 2* np.pi * q * edge_attr[:, 1])

    if normalization is None:
        # L = D_bar_sym - A_sym Hadamard \exp(i \Theta^{(q)}).
        edge_index, _ = add_self_loops(edge_index_sym, num_nodes=num_nodes)
        edge_weight = torch.cat([-edge_weight_sym * edge_weight_q, deg], dim=0)
    elif normalization == 'sym':
        # Compute A_norm = D_bar_sym^{-1/2} A_sym D_bar_sym^{-1/2} Hadamard \exp(i \Theta^{(q)}).
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight_sym * deg_inv_sqrt[col] * edge_weight_q

        # L = I - A_norm.
        edge_index, tmp = add_self_loops(edge_index_sym, -edge_weight,
                                         fill_value=1., num_nodes=num_nodes)
        assert tmp is not None
        edge_weight = tmp
    if not return_lambda_max:
        return edge_index, edge_weight.real, edge_weight.imag
    else:
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        lambda_max = eigsh(L, k=1, which='LM', return_eigenvectors=False)
        lambda_max = float(lambda_max.real)
        return edge_index, edge_weight.real, edge_weight.imag, lambda_max



class MSConv(MessagePassing):
    r"""Magnetic Signed Laplacian Convolution Layer.
    
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size :math:`K`.
        q (float, optional): Initial value of the phase parameter, 0 <= q <= 0.25. Default: 0.25.
        trainable_q (bool, optional): whether to set q to be trainable or not. (default: :obj:`False`)
        normalization (str, optional): The normalization scheme for the magnetic
            Laplacian (default: :obj:`sym`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \bar{\mathbf{D}} - \mathbf{A} \odot \exp(i \Theta^{(q)})`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \bar{\mathbf{D}}^{-1/2} \mathbf{A}
            \bar{\mathbf{D}}^{-1/2} \odot \exp(i \Theta^{(q)})`
            `\odot` denotes the element-wise multiplication.
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the __norm__ matrix on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        absolute_degree (bool, optional): Whether to calculate the degree matrix with respect to absolute entries of the adjacency matrix. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels:int, out_channels:int, K:int, q:float, trainable_q:bool,
                 normalization:str='sym', bias:bool=True, cached: bool=False, absolute_degree: bool=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(MSConv, self).__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym'], 'Invalid normalization'
        kwargs.setdefault('flow', 'target_to_source')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.cached = cached
        self.trainable_q = trainable_q
        self.absolute_degree = absolute_degree

        if trainable_q:
            self.q = Parameter(torch.Tensor(1).fill_(q))
        else:
            self.q = q
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None
        self.cached_q = None

    def __norm__(
        self,
        edge_index,
        num_nodes: Optional[int],
        edge_weight: OptTensor,
        q: float, 
        normalization: Optional[str],
        lambda_max,
        dtype: Optional[int] = None
    ):
        """
        Get the magnetic signed Laplacian.
        
        Arg types:
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * num_nodes (int, Optional) - Node features.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
            * lambda_max (optional, but mandatory if normalization is None) - Largest eigenvalue of Laplacian.
        Return types:
            * edge_index_real, edge_index_imag, edge_weight_real, edge_weight_imag (PyTorch Float Tensor) - signed directed laplacian tensor: real and imaginary edge indices and weights.
        """
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight_real, edge_weight_imag = get_magnetic_signed_Laplacian(
            edge_index, edge_weight, normalization, dtype, num_nodes, q, absolute_degree=self.absolute_degree)
        edge_weight_real = edge_weight_real.to(lambda_max.device)
        edge_weight_imag = edge_weight_imag.to(lambda_max.device)
        edge_weight_real = (2.0 * edge_weight_real) / lambda_max
        edge_weight_real.masked_fill_(edge_weight_real == float("inf"), 0)
        edge_index_imag = edge_index.clone()

        edge_index_real, edge_weight_real = add_self_loops(
            edge_index, edge_weight_real, fill_value=-1.0, num_nodes=num_nodes
        )
        assert edge_weight_real is not None

        edge_weight_imag = (2.0 * edge_weight_imag) / lambda_max
        edge_weight_imag.masked_fill_(edge_weight_imag == float("inf"), 0)

        assert edge_weight_imag is not None

        return edge_index_real.to(lambda_max.device), edge_index_imag.to(lambda_max.device), edge_weight_real.to(lambda_max.device), edge_weight_imag.to(lambda_max.device)

    def forward(
        self,
        x_real: torch.FloatTensor, 
        x_imag: torch.FloatTensor, 
        edge_index: torch.LongTensor,
        edge_weight: OptTensor = None,
        lambda_max: OptTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the Signed Directed Magnetic Laplacian Convolution layer.
        
        Arg types:
            * x_real, x_imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
            * lambda_max (optional, but mandatory if normalization is None) - Largest eigenvalue of Laplacian.
        Return types:
            * out_real, out_imag (PyTorch Float Tensor) - Hidden state tensor for all nodes, with shape (N_nodes, F_out).
        """
        if self.trainable_q:
            self.q = Parameter(torch.clamp(self.q, 0, 0.25))

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))
            if self.q != self.cached_q:
                raise RuntimeError(
                    'Cached q is {}, but found {} in input. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_q, self.q))
        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.trainable_q:
                self.cached_q = self.q.detach().item()
            else:
                self.cached_q = self.q
            if self.normalization != 'sym' and lambda_max is None:
                if self.trainable_q:
                    raise RuntimeError('Cannot train q while not calculating maximum eigenvalue of Laplacian!')
                _, _, _, lambda_max =  get_magnetic_signed_Laplacian(
                edge_index, edge_weight, None, q=self.q, return_lambda_max=True, absolute_degree=self.absolute_degree
            )

            if lambda_max is None:
                lambda_max = torch.tensor(2.0, dtype=x_real.dtype, device=x_real.device)
            if not isinstance(lambda_max, torch.Tensor):
                lambda_max = torch.tensor(lambda_max, dtype=x_real.dtype,
                                        device=x_real.device)
            assert lambda_max is not None
            edge_index_real, edge_index_imag, norm_real, norm_imag = self.__norm__(edge_index, x_real.size(self.node_dim),
                                         edge_weight, self.q, self.normalization,
                                         lambda_max, dtype=x_real.dtype)
            self.cached_result = edge_index_real, edge_index_imag, norm_real, norm_imag

        edge_index_real, edge_index_imag, norm_real, norm_imag = self.cached_result

        Tx_0_real_real = x_real
        Tx_0_imag_imag = x_imag
        Tx_0_imag_real = x_real
        Tx_0_real_imag = x_imag
        out_real_real = torch.matmul(Tx_0_real_real, self.weight[0])
        out_imag_imag = torch.matmul(Tx_0_imag_imag, self.weight[0])
        out_imag_real = torch.matmul(Tx_0_imag_real, self.weight[0])
        out_real_imag = torch.matmul(Tx_0_real_imag, self.weight[0])

        # propagate_type: (x: Tensor, norm: Tensor)
        if self.weight.size(0) > 1:
            Tx_1_real_real = self.propagate(edge_index_real, x=x_real, norm=norm_real, size=None)
            out_real_real = out_real_real + torch.matmul(Tx_1_real_real, self.weight[1])
            Tx_1_imag_imag = self.propagate(edge_index_imag, x=x_imag, norm=norm_imag, size=None)
            out_imag_imag = out_imag_imag + torch.matmul(Tx_1_imag_imag, self.weight[1])
            Tx_1_imag_real = self.propagate(edge_index_real, x=x_real, norm=norm_real, size=None)
            out_imag_real = out_imag_real + torch.matmul(Tx_1_imag_real, self.weight[1])
            Tx_1_real_imag = self.propagate(edge_index_imag, x=x_imag, norm=norm_imag, size=None)
            out_real_imag = out_real_imag + torch.matmul(Tx_1_real_imag, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2_real_real = self.propagate(edge_index_real, x=Tx_1_real_real, norm=norm_real, size=None)
            Tx_2_real_real = 2. * Tx_2_real_real - Tx_0_real_real
            out_real_real = out_real_real + torch.matmul(Tx_2_real_real, self.weight[k])
            Tx_0_real_real, Tx_1_real_real = Tx_1_real_real, Tx_2_real_real

            Tx_2_imag_imag = self.propagate(edge_index_imag, x=Tx_1_imag_imag, norm=norm_imag, size=None)
            Tx_2_imag_imag = 2. * Tx_2_imag_imag - Tx_0_imag_imag
            out_imag_imag = out_imag_imag + torch.matmul(Tx_2_imag_imag, self.weight[k])
            Tx_0_imag_imag, Tx_1_imag_imag = Tx_1_imag_imag, Tx_2_imag_imag

            Tx_2_imag_real = self.propagate(edge_index_real, x=Tx_1_imag_real, norm=norm_real, size=None)
            Tx_2_imag_real = 2. * Tx_2_imag_real - Tx_0_imag_real
            out_imag_real = out_imag_real + torch.matmul(Tx_2_imag_real, self.weight[k])
            Tx_0_imag_real, Tx_1_imag_real = Tx_1_imag_real, Tx_2_imag_real

            Tx_2_real_imag = self.propagate(edge_index_imag, x=Tx_1_real_imag, norm=norm_imag, size=None)
            Tx_2_real_imag = 2. * Tx_2_real_imag - Tx_0_real_imag
            out_real_imag = out_real_imag + torch.matmul(Tx_2_real_imag, self.weight[k])
            Tx_0_real_imag, Tx_1_real_imag = Tx_1_real_imag, Tx_2_real_imag

        out_real = out_real_real - out_imag_imag
        out_imag = out_imag_real + out_real_imag

        if self.bias is not None:
            out_real += self.bias
            out_imag += self.bias

        return out_real, out_imag


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)



class complex_relu_layer(nn.Module):
    """The complex ReLU layer from the `MagNet: A Neural Network for Directed Graphs. <https://arxiv.org/pdf/2102.11391.pdf>`_ paper.
    """
    def __init__(self, ):
        super(complex_relu_layer, self).__init__()
    
    def complex_relu(self, real:torch.FloatTensor, img:torch.FloatTensor):
        """
        Complex ReLU function.
        
        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
        Return types:
            * real, imag (PyTorch Float Tensor) - Node features after complex ReLU.
        """
        mask = 1.0*(real >= 0)
        return mask*real, mask*img

    def forward(self, real:torch.FloatTensor, img:torch.FloatTensor):
        """
        Making a forward pass of the complex ReLU layer.
        
        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
        Return types:
            * real, imag (PyTorch Float Tensor) - Node features after complex ReLU.
        """
        real, img = self.complex_relu(real, img)
        return real, img


class MSGNN_link_prediction(nn.Module):
    r"""The MSGNN model for link prediction.
    
    Args:
        num_features (int): Size of each input sample.
        hidden (int, optional): Number of hidden channels.  Default: 2.
        K (int, optional): Order of the Chebyshev polynomial.  Default: 2.
        q (float, optional): Initial value of the phase parameter, 0 <= q <= 0.25. Default: 0.25.
        label_dim (int, optional): Number of output classes.  Default: 2.
        activation (bool, optional): whether to use activation function or not. (default: :obj:`True`)
        trainable_q (bool, optional): whether to set q to be trainable or not. (default: :obj:`False`)
        layer (int, optional): Number of MSConv layers. Deafult: 2.
        dropout (float, optional): Dropout value. (default: :obj:`0.5`)
        normalization (str, optional): The normalization scheme for the signed directed
            Laplacian (default: :obj:`sym`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \bar{\mathbf{D}} - \mathbf{A} Hadamard \exp(i \Theta^{(q)})`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \bar{\mathbf{D}}^{-1/2} \mathbf{A}
            \bar{\mathbf{D}}^{-1/2} Hadamard \exp(i \Theta^{(q)})`
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the __norm__ matrix on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        absolute_degree (bool, optional): Whether to calculate the degree matrix with respect to absolute entries of the adjacency matrix. (default: :obj:`True`)
    """
    def __init__(self, num_features:int, hidden:int=2, q:float=0.25, K:int=2, label_dim:int=2, \
        activation:bool=True, trainable_q:bool=False, layer:int=2, dropout:float=0.5, normalization:str='sym', cached: bool=False, absolute_degree: bool=True):
        super(MSGNN_link_prediction, self).__init__()

        chebs = nn.ModuleList()
        chebs.append(MSConv(in_channels=num_features, out_channels=hidden, K=K, \
            q=q, trainable_q=trainable_q, normalization=normalization))
        self.normalization = normalization
        self.activation = activation
        if self.activation:
            self.complex_relu = complex_relu_layer()

        for _ in range(1, layer):
            chebs.append(MSConv(in_channels=hidden, out_channels=hidden, K=K,\
                q=q, trainable_q=trainable_q, normalization=normalization, cached=cached, absolute_degree=absolute_degree))

        self.Chebs = chebs
        self.linear = nn.Linear(hidden*4, label_dim)      
        self.dropout = dropout

    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, real: torch.FloatTensor, imag: torch.FloatTensor, edge_index: torch.LongTensor, \
        query_edges: torch.LongTensor, edge_weight: Optional[torch.LongTensor]=None) -> torch.FloatTensor:
        """
        Making a forward pass of the MagNet node classification model.
        
        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * query_edges (PyTorch Long Tensor) - Edge indices for querying labels.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * log_prob (PyTorch Float Tensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
        """
        for cheb in self.Chebs:
            real, imag = cheb(real, imag, edge_index, edge_weight)
            if self.activation:
                real, imag = self.complex_relu(real, imag)

        x = torch.cat((real[query_edges[:,0]], real[query_edges[:,1]], imag[query_edges[:,0]], imag[query_edges[:,1]]), dim = -1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        self.z = x.clone()
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x








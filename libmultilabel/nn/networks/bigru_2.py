import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..networks.base import BaseModel

import torch.nn.functional as F
import torch_geometric.transforms as T
from scipy import sparse
from torch_geometric.nn import GCNConv
import numpy as np


import pickle as pk 

def get_gcn_data(file_name):

  with open(file_name, 'rb') as f:
    data = pk.load(f)

  with open('use_use_m_None_2.pk', 'rb') as f:
    data_2 = pk.load(f)
  
  edm = data['emd']
  adj = data_2['edge']

  return edm, adj





from torch_sparse import SparseTensor

class GCN(torch.nn.Module):
    def __init__(self, dim_sim, out_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dim_sim, 1024)
        self.conv2 = GCNConv(1024, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return x


class BiGRU(BaseModel):
    """BiGRU (Bidirectional Gated Recurrent Unit)
    Args:
        embed_vecs (FloatTensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        rnn_dim (int): The size of bidirectional hidden layers. The hidden size of the GRU network
            is set to rnn_dim//2. Defaults to 512.
        rnn_layers (int): Number of recurrent layers. Defaults to 1.
        dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        activation (str): Activation function to be used. Defaults to 'tanh'.
    """
    def __init__(
        self,
        embed_vecs,
        num_classes,
        gcn_file,
        model_mode,
        model_dim_gcn,
        soft_max, 
        line, 
        conca,
        rnn_dim=512,
        rnn_layers=1,
        dropout=0.2,
        activation='tanh',
        **kwargs
    ):
        super(BiGRU, self).__init__(embed_vecs, dropout, activation, **kwargs)
        assert rnn_dim%2 == 0, """`rnn_dim` should be even."""

        # BiGRU
        emb_dim = embed_vecs.shape[1]
        self.rnn = nn.GRU(emb_dim, rnn_dim//2, rnn_layers,
                          bidirectional=True, batch_first=True)
        self.model_dim_gcn = model_dim_gcn

        self.att = Normal_gcn(rnn_dim, num_classes, model_mode, soft_max, line, conca)


        x_da, adj_da          = get_gcn_data(gcn_file)
        self.x_da_f           = torch.nn.Parameter(
            torch.Tensor(x_da).cuda(), requires_grad=True)
        self.A                = torch.Tensor(adj_da).cuda()
        self.edge_index       = self.A.nonzero(as_tuple=False).t()
        self.edge_weight      = torch.nn.Parameter(self.A[self.edge_index[0], self.edge_index[1]],requires_grad=True)
        self.adj              = SparseTensor(row=self.edge_index[0], col=self.edge_index[1], value=self.edge_weight,
                          sparse_sizes=(50,50))
        

        self.gcn             = GCN(self.x_da_f.shape[-1], 1024).cuda()

    def forward(self, input):
        text, length, indices = self.sort_data_by_length(input['text'], input['length'])
        x = self.embedding(text) # (batch_size, length, rnn_dim)
        x = self.embed_drop(x) # (batch_size, length, rnn_dim)

        packed_inputs = pack_padded_sequence(x, length, batch_first=True)
        x, _ = self.rnn(packed_inputs)

        x = pad_packed_sequence(x)[0]
        x = x.permute(1, 0, 2)
        x = torch.tanh(x)

        x_gcn = self.gcn(self.x_da_f, self.adj)
        x = self.att(x, x_gcn)
        
        return {'logits': x[indices]}

    def sort_data_by_length(self, data, length):
        """Sort data by lengths. If data is not sorted before calling `pack_padded_sequence`,
        under `enforce_sorted=False`, the setting of `use_deterministic_algorithms` must be False.
        To keep `use_deterministic_algorithms` True, we sort the input here and return indices for
        restoring the original order.
        Args:
            data (torch.Tensor): Batch of sequences with shape (batch_size, length)
            length (list): List of text lengths before padding.
        Returns:
            data (torch.Tensor): Sequences sorted by lengths in descending order.
            length (torch.Tensor): Lengths sorted in descending order.
            indices (torch.Tensor): The indexes of the elements in the original data.
        """
        length = torch.as_tensor(length, dtype=torch.int64)
        length, sorted_indices = torch.sort(length, descending=True)
        sorted_indices = sorted_indices.to(data.device)
        data = data.index_select(0, sorted_indices)

        data_size = sorted_indices.size(-1)
        indices = torch.empty(data_size, dtype=torch.long)
        indices[sorted_indices] = torch.arange(data_size)

        return data, length, indices






















def compress_output(d, method, kernel_size, 
                    out_channels):
  dim = 2
  if method   == 'amax':
    return torch.amax(d, dim)
  elif method == 'amin':
    return torch.amin(d, dim)
  elif method == 'mean':
    return torch.mean(d, dim)
  elif method == 'norm':
    return torch.norm(d, dim=dim)
  elif method == 'std':
    return torch.std(d, dim=dim)
  elif method == 'sum':
    return torch.sum(d, dim=dim)
  elif method == 'both':
    cov_out = torch.nn.Conv1d(in_channels = d.shape[1], 
                              out_channels= out_channels, 
                              kernel_size = kernel_size, 
                              stride= 1).cuda()(d)
    fin     = torch.nn.MaxPool1d(kernel_size, 
                                 cov_out.shape[-1]).cuda()(
                                         cov_out).squeeze()
    return fin





class Normal_gcn(torch.nn.Module):

    def __init__(self, rnn_dim, num_classes, 
                 mode, soft_max, line, conca):

        super(Normal_gcn, self).__init__()

        self.U = nn.Linear(rnn_dim, num_classes)
        xavier_uniform_(self.U.weight)

        # linear output
        self.final = nn.Linear(rnn_dim, num_classes)
        xavier_uniform_(self.final.weight)

        

        self.mode     = mode
        self.soft_max = int(soft_max)
        self.line     = int(line)
        self.conca    = int(conca)

        if self.line:
          self.lin_2 = nn.Linear(50, 50)

    def forward(self, x, gcn_out):

      if self.soft_max == 1:
        gcn_out = torch.softmax(gcn_out, dim=1)
      

      alpha = torch.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
      m = alpha.matmul(x)      
      x = gcn_out.mul(m)
      x = compress_output(x, self.mode, 3, 50, self.cuda_r)
      x = x.add(self.final.bias)
      if self.line == 1:
        x = self.lin_2(x)

      if self.conca == 1:
        new_g    = gcn_out.view(x.shape[0],-1)
        coat_out = torch.cat((x, new_g),1)
        lin_2    = torch.nn.Linear(coat_out.shape[-1], x.shape[-1])
        x        = lin_2(coat_out)
      
      return x

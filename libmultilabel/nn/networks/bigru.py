import pickle as pk
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..networks.base import BaseModel

import torch.nn.functional as F
import torch_geometric.transforms as T
from scipy import sparse
from torch_geometric.nn import GCNConv,GATConv,SGConv
import numpy as np


def transform3to2d(d, method, dim):
  
  dim = int(dim)
  if method   == 'amax':
    return torch.amax(d, dim)
  elif method == 'amin':
    return torch.amin(d, dim)
  elif method == 'logsum':
    return torch.logsumexp(d, dim)
  elif method == 'mean':
    return torch.mean(d, dim)
#   elif method == 'median':
#     return torch.median(d, dim).values
  elif method == 'norm':
    return torch.norm(d, dim=dim)
  elif method == 'pod':
    return torch.prod(d, dim=dim)
  elif method == 'std':
    return torch.std(d, dim=dim)
  elif method == 'sum':
    return torch.sum(d, dim=dim)
#   elif method == 'flatten':
#     return d.view(d.shape[0], -1)
  elif method == 'last':
    return d[:,-1]

  elif method == 'all':
    print({'amax': torch.amax(d, dim).shape, 'amin': torch.amin(d, dim).shape, 
            'logsum': torch.logsumexp(d, dim).shape, 'mean': torch.mean(d, dim).shape, 
#             'meadin': torch.median(d, dim).values.shape, 
            'norm': torch.norm(d, dim=dim).shape, 'pod': torch.prod(d, dim=dim).shape, 
            'std': torch.std(d, dim=dim).shape, 'sum': torch.sum(d, dim=dim).shape,
           'last': d[:,-1].shape})






def transform_cov(d, method, cuda_ve, kernel_size, out_channel = None):

#   if method == 'maxpool':
#     return torch.nn.MaxPool1d(kernel_size, stride = 1024)(d).squeeze()
#   elif method == 'avgpool':
#     return torch.nn.AvgPool1d(kernel_size, stride = 1024)(d).squeeze()
#   elif method == 'adoptive':
#     return torch.nn.AdaptiveMaxPool1d(kernel_size)(d).view(d.shape[0],-1)

  d = d.to(cuda_ve)
  if method == 'covd1d':
    return torch.nn.Conv1d(in_channels = d.shape[1], out_channels= out_channel, kernel_size = kernel_size, stride=d.shape[-1]).to(cuda_ve)(d).squeeze()
  elif method == 'both':
    cov_out = torch.nn.Conv1d(in_channels = d.shape[1], out_channels= out_channel, kernel_size = kernel_size, stride= 1)(d).to(cuda_ve)
    fin     = torch.nn.MaxPool1d(kernel_size, cov_out.shape[-1]).to(cuda_ve)(cov_out).squeeze()
    return fin
  elif method == 'all':
    cov_out = torch.nn.Conv1d(in_channels = d.shape[1], out_channels= out_channel, kernel_size = kernel_size, stride= 1).to(cuda_ve)(d)
    fin     = torch.nn.MaxPool1d(kernel_size, cov_out.shape[-1])(cov_out).squeeze().to(cuda_ve)


#     drt = {'maxpool': torch.nn.MaxPool1d(kernel_size, stride = d.shape[-1])(d).squeeze().shape, 
#            'avgpool': torch.nn.AvgPool1d(kernel_size, stride = d.shape[-1])(d).squeeze().shape, 
#            'adoptive' : torch.nn.AdaptiveMaxPool1d(kernel_size)(d).view(d.shape[0],-1).shape, 
#            'covd1d': torch.nn.Conv1d(in_channels = d.shape[1], out_channels= out_channel, kernel_size = kernel_size, stride=d.shape[-1]).cuda()(d).squeeze().shape, 
#            'both': fin.shape}
#     print(drt)
    
    drt = {'covd1d': torch.nn.Conv1d(in_channels = d.shape[1], out_channels= out_channel, kernel_size = kernel_size, stride=d.shape[-1]).cuda()(d).squeeze().shape, 
           'both': fin.shape}
    print(drt)

















import pickle as pk 


def get_gcn_data(file_name):

  with open(file_name, 'rb') as f:
    data = pk.load(f)

  with open('data/use_use_m_None_2 (1).pk', 'rb') as f:
    data_2 = pk.load(f)
  
  edm = data['emd']
  adj = data_2['edge']

  return edm, adj


from torch_sparse import SparseTensor

# def get_data_gcn():
#   x_da, adj_da = get_gcn_data('/content/drive/MyDrive/gcn_data_2/elmo_elmo_1024_2.pk')
#   return x_da_f, adj



class GCN(torch.nn.Module):
    def __init__(self, dim_sim, out_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dim_sim, 1024)
        self.conv2 = GCNConv(1024, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return x.t()

# class GCN(torch.nn.Module):
#     def __init__(self):
#         super(GCN, self).__init__()
#         self.conv1 = GATConv(1024, 1024, heads = 2, concat = False)
#         self.conv2 = GATConv(1024, 1024, heads = 2, concat = False)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)

#         return x.t()




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

        # context vectors for computing attention
        self.U = nn.Linear(rnn_dim, num_classes)
        xavier_uniform_(self.U.weight)

        # linear output
        self.final = nn.Linear(rnn_dim, num_classes)
        xavier_uniform_(self.final.weight)
        
        self.model_mode    = model_mode
        self.model_dim_gcn = model_dim_gcn


        x_da, adj_da          = get_gcn_data(gcn_file)
        self.x_da_f           = torch.nn.Parameter(
            torch.Tensor(x_da).to(f'cuda:{self.model_dim_gcn}'), requires_grad=True)
        self.A                = torch.Tensor(adj_da).to(f'cuda:{self.model_dim_gcn}')
        self.edge_index       = self.A.nonzero(as_tuple=False).t()
        self.edge_weight      = torch.nn.Parameter(self.A[self.edge_index[0], self.edge_index[1]],requires_grad=True)
        self.adj              = SparseTensor(row=self.edge_index[0], col=self.edge_index[1], value=self.edge_weight,
                          sparse_sizes=(50,50))
        

        self.gcn             = GCN(self.x_da_f.shape[-1], 1024).to(f'cuda:{self.model_dim_gcn}')

    def forward(self, input):
        x = self.embedding(input['text']) # (batch_size, length, rnn_dim)
        x = self.embed_drop(x) # (batch_size, length, rnn_dim)

        x, _ = self.rnn(x)
        x    = torch.tanh(x)
        
#         cm = transform_cov(x, 'all', 3, 1024)
        
        if str(self.model_mode) in ['covd1d', 'both']:
            x = transform_cov(x, str(self.model_mode), f'cuda:{self.model_dim_gcn}', 3, 1024)
        else:
            x = transform3to2d(x, str(self.model_mode), 1)
        
        x_gcn = self.gcn(self.x_da_f, self.adj)
        x     = torch.matmul(x, x_gcn)

        return {'logits': x}
    
    
# python3 main.py --config example_config/MIMIC-50/bigru.yml --train_path data/MIMIC-50/train.txt --test_path data/MIMIC-50/test.txt --val_path data/MIMIC-50/test.txt --embed_file /home/admin/Monk/embe_experiments/LibMultiLabel/processed_full.embed --gcn_file /home/admin/Monk/embe_experiments/LibMultiLabel/data/gcn_data_3/w2v_100_sentence.pk --model_attach_mode amax --gcn_dim 1 --gpu_id 0



# all_ops = ['amax', 'amin', 'logsum', 'mean', 'norm', 'pod', 'std', 'sum', 'last']
# cov_ops = ['covd1d', 'both']



# embd_com = ['use_l_None_4.embed', 
#             'use_l_None_5.embed', 
#             'use_m_None_4.embed', 
#             'glove_100_None.embed', 
#             'glove_50_None.embed', 
#             'glove_300_None.embed', 
#             'use_m_None_3.embed', 
#             'use_m_None_2.embed', 
#             'custom']



# gcn_emd = ['w2v_50_sentence.pk', 'w2v_100_sentence.pk', 
#            'w2v_300_sentence.pk', 'w2v_50_word.pk', 
#            'w2v_100_word.pk', 'w2v_100_word.pk']

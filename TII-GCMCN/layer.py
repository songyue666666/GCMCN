from __future__ import division
from turtle import forward
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
from torch.autograd import Variable
from minepy import MINE
import numpy as np


class GCN(nn.Module):
    def __init__(self, cin, cout, dropout):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.W = nn.Parameter(torch.zeros(size=(cin, cout)))  
    
    def forward(self, x, adj):
        d = torch.sqrt(torch.sum(adj, dim=1))
        D = torch.diag(d)
        D_ = D.inverse()
        adj_ = D - adj
        L = torch.mul(torch.mul(D_, adj_), D_)
        x = torch.einsum('cc, abcd->abcd',(L, x))
        x = torch.einsum('abcd, be->aecd',(x, self.W))
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class attention_dilated_inception(nn.Module):    # the dilation convolution block ——by S.Y. 
    def __init__(self, cin, cout, max_kernelsize=5, dilation_factor=2):
        super(attention_dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        # self.kernel_set = [1 ,3, 5]
        self.kernel_set = [i for i in range(1, max_kernelsize+1)] # this step have a great impact on the training speed
        # cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))
        self.W = nn.Parameter(torch.zeros(size=(len(self.kernel_set)*cout, cout))) # self-attention

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        x = torch.einsum('abcd, be->aecd',(x, self.W))
        return x


class embed_conv(nn.Module):
    def __init__(self):
        super(embed_conv,self).__init__()

    def forward(self, x, embed):
        x = torch.einsum('abcd,eb->aecd',(x, embed)) # matrix mutiple
        return x.contiguous() # Take the tensor and put it in a continuous distribution in memory


class graph_construct_mic(nn.Module):   # the graph construction block ——by S.Y. 
    def __init__(self, nnodes, threshold, dim):
        super(graph_construct_mic, self).__init__()

        self.nnodes = nnodes
        self.embed_conv = embed_conv()
        # node embedding
        self.emb = nn.Embedding(nnodes, dim)
        # linear transform
        self.lin = nn.Linear(dim, dim)

        self.threshold = threshold
        self.dim = dim
        

    def forward(self, x, idx):
        nodevec = self.emb(idx)
        nodevec = torch.tanh(self.lin(nodevec))
        lin = nodevec.detach().numpy()
        mine = MINE(alpha=0.6, c=15)
        mic_mat = np.zeros((self.nnodes, self.nnodes)) # calculate the mic correlations
        for i in range(self.nnodes):
            for j in range(self.nnodes):
                mine.compute_score(lin[i], lin[j])
                mic_mat[i, j] = mic_mat[j, i] = mine.mic()
        # adj = F.softmax(F.relu(torch.from_numpy(mic_mat)), dim=1).to(torch.float32) # softmax remove
        adj = F.relu(torch.from_numpy(mic_mat)).to(torch.float32) # obtain the adjacent matrix A
        zero = torch.zeros_like(adj)
        adj = torch.where(adj < self.threshold, zero, adj) # make A sparse by setting the threshold
        nodevec = torch.transpose(nodevec, 0, 1)
        x = self.embed_conv(x, nodevec) # use the node embeddings as an attention matrix for x
        return adj, x


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

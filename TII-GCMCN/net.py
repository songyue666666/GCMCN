from layer import *


class gcmcn(nn.Module):
    def __init__(self, gcn_true, buildA_true, num_nodes, device, predefined_A=None, dropout=0.3, adj_threshold=0.8, node_dim=40, dilation_exponential=2, conv_channels=32, end_channels=128, seq_length=50, in_dim=2, out_dim=12, layers=3, max_kernel_size=5, layer_norm_affline=True):
        super(gcmcn, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.seq_length = seq_length
        self.multiresolution_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.norm = nn.ModuleList()
        # extend the dimension of x.
        # x: (batch, nums_node, sequence_lenth) -> (batch, num_nodes/node_dim, nums_node, sequence_length)
        if self.gcn_true:
            self.start_conv = nn.Conv2d(in_channels=in_dim,out_channels=self.num_nodes, kernel_size=(1, 1))
            # creat the graph. obtain the adjacent matrix and the processed x. 
            # x: (batch, 1, nums_node, sequence_lenth) -> (batch, nodeembedding_length, nums_node, sequence_length)
            self.gc = graph_construct_mic(self.num_nodes, adj_threshold, self.node_dim)
        else:
            self.start_conv = nn.Conv2d(in_channels=in_dim,out_channels=self.node_dim, kernel_size=(1, 1))
        

        kernel_size = max_kernel_size # the max size of kernels(have a great impact on the training time)
        if dilation_exponential>1: # calculate the size of receptive field
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        new_dilation = 1 # record the dilation factor
        for j in range(1,layers+1):
            # calculate the size of layer normalization
            if dilation_exponential>1:
                rf_size_j = int(1+(kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
            else:
                rf_size_j = 1+j*(kernel_size-1)

            self.multiresolution_convs.append(attention_dilated_inception(self.node_dim, conv_channels, max_kernelsize = kernel_size, dilation_factor=new_dilation))
            # if no gcn, change for residual convolution 
            self.residual_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=self.node_dim, kernel_size=(1, 1)))

            if self.gcn_true:
                self.gconv.append(GCN(conv_channels, self.node_dim, dropout))

            if self.seq_length>self.receptive_field:
                self.norm.append(LayerNorm((self.node_dim, self.num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
            else:
                self.norm.append(LayerNorm((self.node_dim, self.num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

            new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=self.node_dim, out_channels=end_channels, kernel_size=(1,1), bias=True)
        if self.seq_length > self.receptive_field:
            self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1,self.seq_length-self.receptive_field+1), bias=True)
        else:
            self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1,1), bias=True)

        self.idx = torch.arange(self.num_nodes).to(device)


    def forward(self, input, idx=None):
        seq_len = input.size(3)   #input: [batch_size, 1, node_nums, seq_in_Len]
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field: # padding first
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))

        x = self.start_conv(input)
        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adj, x = self.gc(x, self.idx)
                else:
                    adj, x = self.gc(x, idx)
            else:
                adj = self.predefined_A

        for i in range(self.layers):
            residual = x
            x = self.multiresolution_convs[i](x)
            x = torch.sigmoid(x)
            # x = torch.tanh(x)
            x = F.dropout(x, self.dropout, training=self.training)
            
            if self.gcn_true:
                x = self.gconv[i](x, adj)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)
        
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn.resolver import activation_resolver
from .gnn import PNA
import math
from performer_pytorch import Performer


class NiNoModel(nn.Module):

    """
    NiNo model for predicting future parameters.
    Default arguments are set for our best performing NiNo model.
    """

    def __init__(self,
                 ctx=5,
                 hid=128,
                 layers=3,
                 gnn=True,
                 dms=True,
                 dms_head = 'mlp',
                 positional_encoding = 'learnable',
                 add_edge_lpe = False,
                 dms_transformer_num_heads = 4,
                 dms_mlp_num_layers = 2,
                 dms_gru_num_layers = 2,
                 dms_lstm_num_layers = 2,
                 dms_transformer_num_layers =2,
                 max_feat_size=9,  # assuming max 3x3 conv
                 input_layer='linear',
                 act_name='silu',
                 residual=True,
                 seq_len=40,
                 scale_method='std',
                 improved_graph=True,
                 wte_pos_enc=True,  # ignored for mlp
                 vocab_size=50257,  # 50257 for GPT2, ignored for mlp
                 edge_types=15,
                 lpe=8,  # ignored for mlp
                 chunk_size=10**5,
                 message_passing_device=None,
                 **kwargs):
        super().__init__()
        self.scale_method = scale_method
        self.ctx = ctx
        self.hid = hid
        self.max_feat_size = max_feat_size
        self.dms = dms
        if not self.dms:
            raise NotImplementedError('Only dms=True is supported in this implementation')
        self.residual = residual
        self.seq_len = seq_len
        self.max_feat_size = 1 if max_feat_size is None else max_feat_size
        self.improved_graph = improved_graph
        if not self.improved_graph:
            raise NotImplementedError('Only improved_graph=True is supported in this implementation')
        self.wte_pos_enc = wte_pos_enc
        self.edge_types = edge_types
        self.lpe = lpe
        self.chunk_size = chunk_size
        self.is_mlp = gnn in [False, None, 'None', 'none']
        self.n_msg_layers = None if self.is_mlp else layers

        if self.edge_types > 0:
            self.layer_embed = nn.Embedding(self.edge_types, hid)

        out_dim = seq_len if self.dms else 1
        mlp_kwargs = {'hid_dim': hid, 'n_layers': 2, 'act_name': act_name}
        self.edge_proj = MLP(in_dim=(1 if self.is_mlp else self.max_feat_size) * ctx,
                             **dict(mlp_kwargs, n_layers=1 if input_layer == 'linear' else 2))
        self.mlp_kwargs = mlp_kwargs

        if self.is_mlp:
            self.edge_mlp = MLP(in_dim=hid,
                                out_dim=out_dim,
                                **dict(mlp_kwargs, n_layers=4))
        else:

            if self.wte_pos_enc:
                self.wte_pos_enc_layer = nn.Embedding(vocab_size + 1, hid)  # +1 for dummy token

            if self.lpe or not self.wte_pos_enc or not self.improved_graph:
                self.node_proj = MLP(in_dim=max(1, self.lpe + int(1 - self.improved_graph) * self.max_feat_size * ctx),
                                     **dict(mlp_kwargs, n_layers=1 if input_layer == 'linear' else 2))

            final_edge_update = kwargs.pop('final_edge_update', False)
            self.gnn = PNA(in_channels=hid,
                           hidden_channels=hid,
                           num_layers=self.n_msg_layers,
                           out_channels=hid,
                           act=act_name,
                           aggregators=['mean'],
                           scalers=['identity'],
                           update_edge_attr=True,
                           modulate_edges=True,
                           gating_edges=False,
                           final_edge_update=final_edge_update,
                           edge_dim=hid,
                           norm=None,
                           chunk_size=chunk_size,
                           message_passing_device=message_passing_device,
                           **kwargs)
            self.add_edge_lpe = add_edge_lpe

            if dms_head == 'mlp':
                print('USING MLP HEAD')
                self.edge_out = MLP(
                    in_dim=hid,
                    out_dim=self.max_feat_size * out_dim,
                    n_layers=dms_mlp_num_layers,  # Use the parameter for number of layers
                    act_name=mlp_kwargs.get('act_name', 'silu')
                )
            elif dms_head == 'gru':
                print('USING GRU HEAD')
                self.edge_out = GRU_DMS(
                    in_dim=hid,
                    hid_dim=hid,
                    out_dim=self.max_feat_size,
                    seq_len=out_dim,  # K steps (sequence length)
                    num_layers=dms_gru_num_layers,
                    dms_mlp_num_layers = dms_mlp_num_layers,  # Use the parameter for number of layers
                    act_name=mlp_kwargs.get('act_name', 'silu'),
                )
            elif dms_head == 'lstm':  # Add support for LSTM
              print('USING LSTM HEAD')
              self.edge_out = LSTM_DMS(
                  in_dim=hid,
                  hid_dim=hid,
                  out_dim=self.max_feat_size,
                  seq_len=out_dim,  # Sequence length
                  num_layers=dms_lstm_num_layers,
                  dms_mlp_num_layers=dms_mlp_num_layers, # Number of layers
                  act_name=mlp_kwargs.get('act_name', 'silu')
              )
            elif dms_head == 'transformer':
              print('USING TRANSFORMER HEAD')
              lpe_contribution = 2 * self.lpe  # LPE contributes this much to edge_attr

              self.edge_out = Transformer_DMS(
                  in_dim=hid + lpe_contribution if add_edge_lpe else hid,
                  hid_dim=hid * 2,  # Feedforward hidden dimension
                  out_dim=self.max_feat_size,
                  num_heads=dms_transformer_num_heads,
                  num_layers=dms_transformer_num_layers,
                  seq_len=seq_len,
                  act_name=mlp_kwargs.get('act_name', 'silu')
              )
            else:
                raise ValueError(f"Unsupported DMS head type: {dms_head}")



            if not self.improved_graph:
                self.node_out = MLP(in_dim=hid,
                                    out_dim=self.max_feat_size * out_dim,
                                    **mlp_kwargs)

        self.initializer_range = 0.02
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, pyg.nn.dense.linear.Linear)):
            # Xavier initialization for linear layers
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # Normal initialization for embedding weights
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                                pyg.nn.norm.layer_norm.LayerNorm, pyg.nn.norm.batch_norm.BatchNorm)):
            # Constant initialization for normalization layers
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.GRU):
            # Orthogonal initialization for GRU weights
            for param_name, param in module.named_parameters():
                if "weight" in param_name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in param_name:
                    param.data.zero_()
        elif isinstance(module, nn.LSTM):
            # Orthogonal initialization for LSTM weights
            for name, param in module.named_parameters():
                if "weight_ih" in name:  # Input-hidden weights
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:  # Hidden-hidden weights
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.zero_()
                    # Forget gate bias initialization
                    n = param.size(0) // 4
                    param.data[n:2 * n].fill_(1.0)
        elif isinstance(module, nn.TransformerEncoderLayer):
            # Default initialization for TransformerEncoderLayer
            for name, param in module.named_parameters():
                if param.dim() > 1:  # Weights
                    nn.init.xavier_uniform_(param.data)
                else:  # Biases
                    param.data.zero_()
        else:
            assert not hasattr(module, 'weight') and not hasattr(module, 'bias'), type(module)

#############################################################

    def fw_split(self, module, x, inplace=False):
        """
        Forward pass of the model with splitting of the input tensor for better memory efficiency.
        :param module: nn.Module
        :param x: input tensor with the first dimension to be sliced
        :param inplace: whether to perform the operation inplace
        :return: output tensor
        """
        if self.training:
            return module(x)
        if not inplace:
            x_out = []
        chunk_size = len(x) if self.chunk_size in [0, -1, None] else self.chunk_size
        for i in range(0, len(x), chunk_size):
            if inplace:
                x[i:i + chunk_size] = module(x[i:i + chunk_size])
            else:
                x_out.append(module(x[i:i + chunk_size]))
        return x if inplace else torch.cat(x_out, dim=0)

    def forward(self, graphs, k=None):
        """
        Forward pass of the model.
        :param graphs: pytorch geometric batch of graphs (can be multiple models stacked as disconnected graphs)
        :param k: number of steps to predict into the future (only used for dms during inference)
        :return: graphs with updated edge (and node) features
        """

        graphs.edge_attr = graphs.edge_attr.unflatten(1, (-1, self.ctx))
        max_feat_size = graphs.edge_attr.shape[1]
        if self.residual:
            edge_attr_res = graphs.edge_attr[:, :, self.ctx - 1]  # last parameter values

        if self.is_mlp:

            # By using chunking in th MLP, we avoid storing the full edge_attr tensor (n_params, feat_dim) in memory
            chunk_size = len(graphs.edge_attr) if self.chunk_size in [0, -1, None] else self.chunk_size

            if self.dms and not self.training:
                assert k is not None and k >= 1, k
                # avoid predicting for all the future steps during inference to save computation
                # only predict for k-th step
                fc = self.edge_mlp.fc
                w = fc[-1].weight.data.clone()
                b = fc[-1].bias.data.clone()
                fc[-1].weight.data = fc[-1].weight.data[k - 1:k]
                fc[-1].bias.data = fc[-1].bias.data[k - 1]
                fc[-1].out_features = 1
            else:
                fc = self.edge_mlp.fc

            device = next(fc[0].parameters()).device
            for i in range(0, len(graphs.edge_attr), chunk_size):
                graphs.edge_attr[i:i + chunk_size, :, :1] = self.fw_split(fc, self.edge_proj(
                    graphs.edge_attr[i:i + chunk_size].to(device)) + (
                    self.layer_embed(graphs.edge_type[i:i + chunk_size].to(device)).unsqueeze(1)
                    if self.edge_types else 0)).to(graphs.edge_attr.device)
            graphs.edge_attr = graphs.edge_attr[:, :, :1]

            if self.dms and not self.training:
                fc[-1].weight.data = w
                fc[-1].bias.data = b

        else:
            edge_types = self.layer_embed(graphs.edge_type) if self.edge_types else 0

            x_lpe = self.fw_split(self.node_proj, graphs.pos) if self.lpe else 0
            wte_pos_emb = self.wte_pos_enc_layer(graphs.pos_w) if self.wte_pos_enc else 0
            if self.lpe:
                assert x_lpe.dim() == 2, x_lpe.shape
            if self.wte_pos_enc:
                assert wte_pos_emb.dim() == 2, wte_pos_emb.shape
            graphs.x = wte_pos_emb + x_lpe
            graphs.edge_attr = graphs.edge_attr.flatten(1, 2)
            assert graphs.x.dim() == graphs.edge_attr.dim() == 2, (graphs.x.shape, graphs.edge_attr.shape)

            dtype = next(self.edge_proj.parameters()).dtype

            if self.training:
                graphs.edge_attr = edge_types + self.edge_proj(graphs.edge_attr.to(dtype))
            else:

                if max_feat_size < self.max_feat_size:
                    fc = self.edge_proj.fc
                    fc[0].weight.data = fc[0].weight.data[:, :max_feat_size * self.ctx]
                    fc[0].in_features = max_feat_size * self.ctx

                chunk_size = len(graphs.edge_attr) if self.chunk_size in [0, -1, None] else self.chunk_size
                if self.edge_types:
                    for i in range(0, len(graphs.edge_attr), chunk_size):
                        edge_types[i:i + chunk_size] = self.edge_proj(
                            graphs.edge_attr[i:i + chunk_size]) + edge_types[i:i + chunk_size]
                else:
                    edge_types = self.edge_proj(graphs.edge_attr)

                graphs.edge_attr = edge_types
                del edge_types

            graphs.x, graphs.edge_attr = self.gnn(
                x=graphs.x, edge_index=graphs.edge_index, edge_attr=graphs.edge_attr)
         

            if self.add_edge_lpe:
                node_embeddings = graphs.x  # Shape: (num_nodes, hidden_dim)

                # Extract edge indices
                edge_index = graphs.edge_index  # Shape: (2, num_edges)
                source_nodes = edge_index[0]  # Source nodes of edges
                target_nodes = edge_index[1]  # Target nodes of edges

                # Get LPE from node embeddings for source and target nodes
                lpe_source = node_embeddings[source_nodes, :self.lpe]  # Assuming first self.lpe dimensions are LPE
                lpe_target = node_embeddings[target_nodes, :self.lpe]

                # Compute edge-level LPE
                edge_lpe = torch.cat([lpe_source, lpe_target], dim=-1)  # Concatenate source and target LPE
                # Alternative: edge_lpe = lpe_source + lpe_target  # Summing LPE of source and target nodes

                # Integrate edge-level LPE into edge attributes
                graphs.edge_attr = torch.cat([graphs.edge_attr, edge_lpe], dim=-1)  

            if self.dms and not self.training:
                assert k is not None and k >= 1, k
                if isinstance(self.edge_out, MLP):
                    # Handle MLP case
                    fc = self.edge_out.fc
                    _, in_dim = fc[-1].weight.shape
                    n_out = max_feat_size
                    w = fc[-1].weight.data.clone()
                    b = fc[-1].bias.data.clone()

                    fc[-1].weight.data = fc[-1].weight.data.reshape(self.max_feat_size, -1, in_dim)[:n_out, k - 1]
                    fc[-1].bias.data = fc[-1].bias.data.reshape(self.max_feat_size, -1)[:n_out, k - 1]
                    fc[-1].out_features = n_out

                    graphs.edge_attr = self.fw_split(fc, graphs.edge_attr).unsqueeze(2)
                    fc[-1].weight.data = w
                    fc[-1].bias.data = b
                elif isinstance(self.edge_out, GRU_DMS):
                    # Access the final Linear layer in the MLP inside GRU_DMS
                    final_fc = self.edge_out.mlp.fc[-1]  # Access the last layer in the Sequential container
                    _, in_dim = final_fc.weight.shape  # Get input dimension of the final Linear layer
                    n_out = max_feat_size

                    # Clone weight and bias for modification
                    w = final_fc.weight.data.clone()
                    b = final_fc.bias.data.clone()

                    # Reshape weight and bias to focus on the k-th step for inference
                    final_fc.weight.data = final_fc.weight.data.reshape(self.max_feat_size, -1, in_dim)[:n_out, k - 1]
                    final_fc.bias.data = final_fc.bias.data.reshape(self.max_feat_size, -1)[:n_out, k - 1]
                    final_fc.out_features = n_out

                    # Process edge attributes using the modified final Linear layer
                    graphs.edge_attr = self.fw_split(final_fc, graphs.edge_attr).unsqueeze(2)

                    # Restore original weight and bias
                    final_fc.weight.data = w
                    final_fc.bias.data = b

                    
                elif isinstance(self.edge_out, LSTM_DMS):
                    # Access the final Linear layer in the MLP inside LSTM_DMS
                    final_fc = self.edge_out.mlp.fc[-1]  # Access the last layer in the Sequential container
                    _, in_dim = final_fc.weight.shape  # Get input dimension of the final Linear layer
                    n_out = max_feat_size

                    # Clone weight and bias for modification
                    w = final_fc.weight.data.clone()
                    b = final_fc.bias.data.clone()

                    # Reshape weight and bias to focus on the k-th step for inference
                    final_fc.weight.data = final_fc.weight.data.reshape(self.max_feat_size, -1, in_dim)[:n_out, k - 1]
                    final_fc.bias.data = final_fc.bias.data.reshape(self.max_feat_size, -1)[:n_out, k - 1]
                    final_fc.out_features = n_out

                    # Process edge attributes using the modified final Linear layer
                    graphs.edge_attr = self.fw_split(final_fc, graphs.edge_attr).unsqueeze(2)

                    # Restore original weight and bias
                    final_fc.weight.data = w
                    final_fc.bias.data = b
                elif isinstance(self.edge_out, Transformer_DMS):
                    # Handle Transformer case
                    # Clone the weights and biases of the final MLP in Transformer_DMS
                    final_fc = self.edge_out.mlp.fc[-1]  # Access the last layer in the Sequential container
                    _, in_dim = final_fc.weight.shape  # Get input dimension of the final Linear layer
                    n_out = max_feat_size

                    # Clone weights and biases for temporary modification
                    w = final_fc.weight.data.clone()
                    b = final_fc.bias.data.clone()

                    # Reshape weights and biases to focus on the k-th step for inference
                    final_fc.weight.data = final_fc.weight.data.reshape(self.max_feat_size, -1, in_dim)[:n_out, k - 1]
                    final_fc.bias.data = final_fc.bias.data.reshape(self.max_feat_size, -1)[:n_out, k - 1]
                    final_fc.out_features = n_out

                    # Process edge attributes using the Transformer and final Linear layer
                    transformed_attr = self.edge_out.transformer(graphs.edge_attr.unsqueeze(1))  # Shape: (num_edges, 1, in_dim)
                    transformed_attr = transformed_attr.squeeze(1)  # Remove batch dimension
                    graphs.edge_attr = self.fw_split(final_fc, transformed_attr).unsqueeze(2)

                    # Restore original weights and biases
                    final_fc.weight.data = w
                    final_fc.bias.data = b


                else:
                    raise ValueError(f"Unsupported DMS head type: {type(self.edge_out)}")

            else:
                graphs.edge_attr = self.fw_split(self.edge_out, graphs.edge_attr).unflatten(
                    1, (self.max_feat_size, -1))
        if self.residual:
            graphs.edge_attr = edge_attr_res.unsqueeze(-1) + graphs.edge_attr

        if self.training:
            graphs.edge_attr = graphs.edge_attr.flatten(1, 2)

        return graphs


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim=None, out_dim=None, n_layers=1, act_name='silu'):
        super().__init__()
        hid_dim = hid_dim or in_dim
        out_dim = out_dim or hid_dim
        layers = []
        for layer in range(n_layers):
            in_dim_ = in_dim if layer == 0 else hid_dim
            out_dim_ = out_dim if layer == n_layers - 1 else hid_dim
            layers.append(nn.Linear(in_dim_, out_dim_))
            if layer < n_layers - 1:
                layers.append(activation_resolver(act_name))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

class GRU_DMS(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, seq_len, num_layers=2,dms_mlp_num_layers=2, act_name='silu'):
        super().__init__()
        self.seq_len = seq_len
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hid_dim, num_layers=num_layers, batch_first=True)
        self.mlp= MLP(
                    in_dim=hid_dim,
                    out_dim=seq_len* out_dim,
                    n_layers=dms_mlp_num_layers, 
                    act_name=act_name
                )   
        self.activation = activation_resolver(act_name)  

    def forward(self, x):
        """
        Forward pass for GRU_DMS.
        :param x: Input tensor of shape (batch_size, num_edges, in_dim)
        :return: Output tensor of shape (batch_size, num_edges, out_dim * seq_len)
        """
        gru_output, _ = self.gru(x)

        out = self.mlp(gru_output)  

        return out

class LSTM_DMS(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, seq_len, num_layers=2,dms_mlp_num_layers=2, act_name='silu'):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hid_dim, num_layers=num_layers, batch_first=True)
        self.mlp = MLP(
            in_dim=hid_dim,
            out_dim=seq_len * out_dim,
            n_layers=dms_mlp_num_layers, 
            act_name=act_name
        )
        self.activation = activation_resolver(act_name) 

    def forward(self, x):
        """
        Forward pass for LSTM_DMS.
        :param x: Input tensor of shape (batch_size, num_edges, in_dim)
        :return: Output tensor of shape (batch_size, num_edges, out_dim * seq_len)
        """
        lstm_output, _ = self.lstm(x)  # LSTM processes the input sequence
        out = self.mlp(lstm_output)  # MLP processes the LSTM output
        return out


import torch.nn as nn

class Transformer_DMS(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_heads, num_layers, seq_len, act_name='silu'):
        """
        Transformer-based DMS class for processing enriched edge embeddings.

        :param in_dim: Input dimension of edge attributes.
        :param hid_dim: Hidden dimension of the transformer.
        :param out_dim: Output dimension for multi-step predictions.
        :param num_heads: Number of attention heads in the transformer.
        :param num_layers: Number of transformer layers.
        :param seq_len: Number of future steps (sequence length for predictions).
        :param act_name: Activation function name (e.g., 'silu').
        """
        super().__init__()
        self.seq_len = seq_len

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,  # Input dimension
            nhead=num_heads,
            dim_feedforward=hid_dim,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.transformer = Performer(
        #     dim=in_dim,
        #     dim_head=in_dim // num_heads,
        #     depth=num_layers,
        #     heads=num_heads,
        #     causal=False  # Set to True if attention should be causal
        # )

        # MLP for multi-step predictions
        self.mlp = MLP(
            in_dim=in_dim,
            out_dim=seq_len * out_dim,
            n_layers=2,
            act_name=act_name
        )

    def forward(self, edge_attr):
        """
        Forward pass for Transformer_DMS.

        :param edge_attr: Enriched edge attributes of shape (num_edges, in_dim).
        :return: Predicted edge attributes of shape (num_edges, seq_len * out_dim).
        """
        edge_attr = edge_attr.unsqueeze(1)  # Shape: (num_edges, 1, in_dim)

        edge_attr = self.transformer(edge_attr)  # Shape: (num_edges, 1, in_dim)

        edge_attr = edge_attr.squeeze(1)  # Shape: (num_edges, in_dim)
        out = self.mlp(edge_attr)  # Shape: (num_edges, seq_len * out_dim)

        return out


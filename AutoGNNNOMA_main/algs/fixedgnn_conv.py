import torch as th
import torch.nn as nn
import numpy as np
from copy import deepcopy
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
import math

th.autograd.set_detect_anomaly(True)

class IGConv(nn.Module):
    def __init__(self, conv1, mlp1, mlp2, num_agent, edge_idx, layer_idx, args):
        super(IGConv, self).__init__()
        self.args = args
        self.conv1 = conv1
        self.mlp1 = mlp1 # local embedding function
        self.mlp2 = mlp2 # combination function
        self.num_agent = num_agent
        self.edge_idx = edge_idx # 记录所有links的tx-rx对，[tx_node_id,rx_node_id]; 对每个batch都一样
        self.num_edge = len(edge_idx)
        # self.link_from_agent = [None for i in range(self.num_agent)]  # 源节点为 i 的links的索引
        self.link_to_agent = [None for i in range(self.num_agent)]  # 目的节点为 i 的links的索引
        for m in range(self.num_agent):
            # self.link_from_agent[m] = (np.argwhere(self.edge_idx[:, 0] == m)).view(-1)
            self.link_to_agent[m] = (np.argwhere(self.edge_idx[:, 1] == m)).view(-1)
        self.layer_idx = layer_idx # index of GNN layer

    def forward(self, x0, x_last, edge_attr):
        """
        # x_last: tensor, [batchsize, num_agent, node_feature_size]
        # e_last: tensor, [batchsize, num_edge, embedding_size]
        # edge_attr: [batchsize, num_links, node_feature_dim]
        """
        # 该函数实现一层GNN layer
        # 所有的逻辑代码都在forward()里面，当调用propagate()函数之后，它将会在内部调用local_embedding()，生成局部embedding并进行聚合
        agg_embedding = self.propagate(x0, x_last, edge_attr)  # Start propagating messages
        return self.update_embedding(agg_embedding, x0, x_last)

    def propagate(self, x0, x_last, edge_attr):
        bs = x0.shape[0] # batch size
        node_feature_j = x0[:, self.edge_idx[:, 0], :]  # [bs, num_links, -1] local state of tx nodes (inter-agent links)
        if self.layer_idx > 1:
            local_state_j = x_last[:, self.edge_idx[:, 0], :]
            local_embed = self.local_embedding(bs, node_feature_j, local_state_j, edge_attr)
        else:
            local_embed = self.local_embedding(bs, node_feature_j, None, edge_attr)
        # 传递局部embedding给每个用户
        agg_in = th.zeros([bs, self.num_agent, local_embed.shape[-1]], device=x0.device)
        for i in range(self.num_agent):
            agg_in[:, i, :] = th.sum(local_embed[:, self.link_to_agent[i], :].view(bs, self.link_to_agent[i].shape[0], -1), dim=1)
        return agg_in

    def local_embedding(self, batch_size, node_feature_j, local_state_j, edge_attr):
        # 该函数定义对于每个节点对 (xi,xj)，怎样生成信息并进行聚合
        x = th.cat((node_feature_j.view(batch_size*self.num_edge,self.args.num_user,-1),edge_attr.view(batch_size*self.num_edge,self.args.num_user,-1)),dim=1)
        y = th.relu(self.conv1(x.view(batch_size*self.num_edge,2,self.args.num_user,-1)))
        y = y.view(batch_size*self.num_edge,-1)
        if self.layer_idx > 1:
            local_feature = th.cat((y, local_state_j.view(batch_size*self.num_edge,-1)), dim=-1)
        else:
            local_feature = y
        local_embedding = self.mlp1(local_feature).view(batch_size, self.num_edge, -1)  # inter-agent local embedding, use self.mlp1 as embedding function
        return local_embedding

    def update_embedding(self, agg_in, x0, x_last):
        # 聚合信息，并更新每个节点的 hidden state
        bs, num_agent = agg_in.shape[0], agg_in.shape[1]
        # 聚合邻居信息embedding + local node feature
        if self.layer_idx > 1:
            agg = th.cat((x0, x_last, agg_in), dim=-1)  # [bs, num_agent, hidden_state_size+embed_size]
        else:
            agg = th.cat((x0, agg_in), dim=-1)  # [bs, num_agent, hidden_state_size+embed_size]
        # update hidden state of each node, use mlp2 as combination function
        comb = self.mlp2(agg.view(bs * num_agent, -1)).view(bs, num_agent, -1) # [bs,M,hidden_state_size]  # mlp2为combine函数
        return comb


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())  # , BN(channels[i]))
        for i in range(1, len(channels))
    ])


class FixedMPGNN(th.nn.Module):
    def __init__(self, args):
        super(FixedMPGNN, self).__init__()
        # config
        hidden_state_size = args.hidden_state_size
        embedding_size = args.embedding_size
        node_feature_size, edge_feature_size = args.node_feature_size, args.edge_feature_size # the shape of the input
        total_dim_peragent = int(args.NNoutput_size)  # +1
        K, N = args.num_user, args.num_antenna
        self.tau = 10
        self.args = args
        self.alpha_M, self.alpha_E = args.alpha_M, args.alpha_E
        # ============================== MPGNN LAYER 1 ==========================
        # local embedding layers (for the original feature)==========
        self.conv1 = th.nn.Conv2d(2,2,6)
        self.mlp1 = MLP([2*((K-6)+1)*((N*2-6)+1), hidden_state_size, embedding_size])
        # combine local hidden state and aggregated embedding
        self.mlp2 = Seq(*[MLP([embedding_size + node_feature_size, hidden_state_size]), Lin(hidden_state_size, hidden_state_size)])
        # ============================== MPGNN LAYER 2 ==========================
        # local embedding layers (for the hidden states + edge feature)
        self.conv2 = th.nn.Conv2d(2, 2, 6)
        # self.mlp3 = MLP([2*((K-6)+1)*((N*2-6)+1) + hidden_state_size, hidden_state_size, embedding_size])
        self.mlp3 = Seq(*[MLP([2*((K-6)+1)*((N*2-6)+1) + hidden_state_size, hidden_state_size]),
                          Lin(hidden_state_size, embedding_size)])
        # combine local hidden state and aggregated embedding
        self.mlp4 = MLP([embedding_size + node_feature_size + hidden_state_size, hidden_state_size, hidden_state_size])
        # ============================== MPGNN LAYER 3 ==========================
        self.conv3 = th.nn.Conv2d(2, 2, 6)
        # local embedding layers (for the hidden states + edge feature)
        # self.mlp5 = MLP([2*((K-6)+1)*((N*2-6)+1) + hidden_state_size, hidden_state_size, embedding_size])
        self.mlp5 = Seq(*[MLP([2*((K-6)+1)*((N*2-6)+1) + hidden_state_size, hidden_state_size]),
                          Lin(hidden_state_size, embedding_size)])
        # combine local hidden state and aggregated embedding
        self.mlp6 = MLP([embedding_size + node_feature_size + hidden_state_size, hidden_state_size, hidden_state_size])
        # ============================== MPGNN LAYER 4 ==========================
        # local embedding layers (for the hidden states + edge feature)
        self.conv4 = th.nn.Conv2d(2, 2, 6)
        # self.mlp7 = MLP([2*((K-6)+1)*((N*2-6)+1)+ hidden_state_size, hidden_state_size, embedding_size])
        self.mlp7 = Seq(*[MLP([2*((K-6)+1)*((N*2-6)+1) + hidden_state_size, hidden_state_size]),
                          Lin(hidden_state_size, embedding_size)])
        # combine local hidden state and aggregated embedding
        self.mlp8 = Seq(
            *[MLP([embedding_size + node_feature_size + hidden_state_size, hidden_state_size]), Lin(hidden_state_size, hidden_state_size)])

        self.gnn1 = IGConv(self.conv1, self.mlp1, self.mlp2, args.num_agent, args.edge_idx, 1, self.args)
        self.gnn2 = IGConv(self.conv2, self.mlp3, self.mlp4, args.num_agent, args.edge_idx, 2, self.args)
        self.gnn3 = IGConv(self.conv3, self.mlp5, self.mlp6, args.num_agent, args.edge_idx, 3, self.args)
        self.gnn4 = IGConv(self.conv4, self.mlp7, self.mlp8, args.num_agent, args.edge_idx, 4, self.args)

        self.fc = Seq(*[MLP([hidden_state_size + node_feature_size, hidden_state_size]), Lin(hidden_state_size, total_dim_peragent+args.num_user)])


    def forward(self, inputs, train_mode=True):
        """
            # inputs: a graph, {"x": node_feature, "edge_atrr": edge_feature}
            # train_mode: True during training or validation
        """

        def get_gumbel_prob(x_input):
            while True:
                gumbels = - th.empty_like(x_input).exponential_().log()  # sampling
                logits = (x_input.log_softmax(dim=-1) + gumbels) / self.tau
                probs = nn.functional.softmax(logits, dim=-1)  # probabilities for sampling
                """======== for inference ======="""
                index = probs.max(-1, keepdim=True)[1]  # argmax of the final dim
                one_h = th.zeros_like(logits).scatter_(-1, index, 1.0)  # one-hot vector
                """======== for back-propogation/training ======="""
                hardwts = one_h - probs.detach() + probs
                if (
                        (th.isinf(gumbels).any())
                        or (th.isinf(probs).any())
                        or (th.isnan(probs).any())
                ):
                    continue
                else:
                    break
            return hardwts, index

        x0,  edge_attr = inputs['x'], inputs['edge_attr']
        batch_size = x0.shape[0]
        x1 = self.gnn1(x0, x0, edge_attr)
        x2 = self.gnn2(x0, x1, edge_attr)
        if self.args.num_layer == 2:
            x_end = x2
        elif self.args.num_layer == 3:
            x_end = self.gnn3(x0, x2, edge_attr)
        elif self.args.num_layer == 4:
            x3 = self.gnn3(x0, x2, edge_attr)
            x_end = self.gnn4(x0, x3, edge_attr)
        y = self.fc(th.cat([x0.view(batch_size * self.args.num_agent, -1),x_end.view(batch_size * self.args.num_agent, -1)], dim=-1))
        N, K = self.args.num_antenna, self.args.num_user


        power_user = 0.01 + (1-0.01*self.args.num_user) * th.softmax((y[:, :K]).view(y.shape[0], -1), dim=-1)
        BF = th.tanh(y[:, K:(N * K * 2 + K)])
        SIC, _ = get_gumbel_prob(y[:, (K + N * K * 2):].view(y.shape[0], int(K * (K - 1) / 2), 3))
        out = th.cat((power_user, BF, SIC.view(y.shape[0], -1)), dim=-1)  # [batch_size*num_agent, 2*N*K+K*(K-1)/2*3]
        return out

    def get_weights(self):
        """
        We empirically found that partially train the following GNN parameters achieve better results
        """
        xlist = list(self.conv1.parameters()) + list(self.conv2.parameters())
        xlist += list(self.mlp1.parameters()) + list(self.mlp2.parameters()) + list(self.mlp3.parameters()) + list(self.mlp4.parameters())
        xlist += list(self.conv3.parameters()) + list(self.conv4.parameters())
        # xlist += list(self.mlp5.parameters()) + list(self.mlp6.parameters()) + list(self.mlp7.parameters()) + list(self.mlp8.parameters())
        xlist += list(self.fc.parameters())
        # xlist = self.parameters()
        return xlist



    def set_tau(self, tau):
        self.tau = tau

    def get_tau(self):
        return self.tau


    def get_comm_cost(self):
        return ((self.alpha_M[:,0].unsqueeze(dim=-1)) * self.alpha_E[:,:,0]).sum() + self.args.embedding_size

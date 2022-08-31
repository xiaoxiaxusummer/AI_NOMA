import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

th.autograd.set_detect_anomaly(True)

class Auto_IGConv(nn.Module):
    def __init__(self, conv1, mlp1, mlp2, num_agent, edge_idx, layer_idx, args):
        super(Auto_IGConv, self).__init__()
        self.args = args
        self.mlp1 = mlp1 # local embedding function
        self.mlp2 = mlp2 # combination function
        self.conv1 = conv1
        self.num_agent = num_agent
        self.edge_idx = edge_idx # all links，[source_node_id,target_node_id]
        self.num_edge = len(edge_idx)
        self.link_to_agent = [None for i in range(self.num_agent)]  # the edges link to node i
        for m in range(self.num_agent):
            # self.link_from_agent[m] = (np.argwhere(self.edge_idx[:, 0] == m)).view(-1)
            self.link_to_agent[m] = (np.argwhere(self.edge_idx[:, 1] == m)).view(-1)
        self.layer_idx = layer_idx  # index of GNN layer

    def forward(self, x0, x_last, edge_attr, alpha_E, alpha_M, dynamic=True):
        """
        # x0: node feature,  [batchsize, num_agent, node_feature_size]
        # x_last: last hidden state, [batchsize, num_agent, hidden_state_size]
        # edge_attr: edge feature, [batchsize, num_links, edge_feature_size]
        # dynamic: dynamic or fixed layer
        """
        # the following function implement one GNN layer
        # When invoke propagate() function, it will invoke local_embedding() to generate embedding and perform aggregation
        agg_embedding = self.propagate(x0, x_last, edge_attr, alpha_E, dynamic)
        return self.update_embedding(agg_embedding, x0, x_last, alpha_M, dynamic)

    def propagate(self, x_0, x_last, edge_attr, alpha_E, dynamic=True):

        # achieve local embedding from each source node
        bs = x_0.shape[0] # batch size
        node_feature_j = x_0[:, self.edge_idx[:, 0],:]  # local state of tx nodes in each link   [batch_size, num_links, -1]
        if self.layer_idx > 1:
            local_state_j = x_last[:, self.edge_idx[:, 0], :]
            local_embed = self.local_embedding(bs, node_feature_j, local_state_j, edge_attr)
        else:
            local_embed = self.local_embedding(bs, node_feature_j, None, edge_attr)

        # propogate local embedding to each end node
        agg_in = th.zeros([bs, self.num_agent, local_embed.shape[-1]], device=x_0.device)

        for i in range(self.num_agent):
            if dynamic: # with zero padding
                x, _ = th.max(local_embed[:,self.link_to_agent[i],:].view(bs,self.link_to_agent[i].shape[0],-1), dim=1)
                agg_in[:, i, :]  = th.mul(alpha_E[:, 0],  x)
            else:
                agg_in[:,i,:],_ = th.max(local_embed[:, self.link_to_agent[i], :].view( bs, self.link_to_agent[i].shape[0], -1) ,dim=1)
        return agg_in

    def local_embedding(self, batch_size, node_feature_j, local_state_j, edge_attr):
        """"Embed node feature and edge feature"""
        x = th.cat((node_feature_j.view(batch_size * self.num_edge, self.args.num_user, -1),
                    edge_attr.view(batch_size * self.num_edge, self.args.num_user, -1)), dim=1)
        y = th.relu(self.conv1(x.view(batch_size * self.num_edge, 2, self.args.num_user, -1)))
        y = y.view(batch_size * self.num_edge, -1)
        if self.layer_idx > 1:
            local_feature = th.cat((y, local_state_j.view(batch_size*self.num_edge,-1)), dim=-1)
        else:
            local_feature = y
        local_embedding = self.mlp1(local_feature).view(batch_size, self.num_edge, -1)  # inter-agent local embedding, use self.mlp1 as embedding function
        return local_embedding

    def update_embedding(self, agg_in, x0, x_last, alpha_M, dynamic=True):
        # aggregate information and update hidden state
        batch_size, num_agent = agg_in.shape[0], agg_in.shape[1]
        # aggregate embedding from neighbors with local node feature
        if self.layer_idx > 1:
            agg = th.cat((x0, x_last, agg_in), dim=-1)  # [bs, num_agent, hidden_state_size+embed_size]
        else:
            agg = th.cat((x_last, agg_in), dim=-1)  # [bs, num_agent, hidden_state_size+embed_size]

        # update hidden state of each node, use mlp2 as combination function
        comb = self.mlp2(agg.view(batch_size * num_agent, -1)).view(batch_size, num_agent, -1) # [batch_size,M,hidden_state_size]  # mlp2 is the combination function
        if dynamic:
            new_x = alpha_M[0]*comb + alpha_M[1]*x_last
        else:
            new_x = comb
        return new_x


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())  # , BN(channels[i]))
        for i in range(1, len(channels))
    ])


class AutoMPGNN(th.nn.Module):
    def __init__(self, args):
        super(AutoMPGNN, self).__init__()
        # config
        hidden_state_size = args.hidden_state_size
        embedding_size = args.embedding_size
        node_feature_size, edge_feature_size = args.node_feature_size, args.edge_feature_size # the shape of the input
        total_dim_peragent = int(args.NNoutput_size)
        K, N = args.num_user, args.num_antenna
        self.tau = 10
        self.args = args
        # ============================== MPGNN LAYER 1 ==========================
        # local embedding layers (for the original feature)==========
        self.conv1 = th.nn.Conv2d(2, 2, 6)
        self.mlp1 = MLP([2 * ((K - 6) + 1) * ((N * 2 - 6) + 1), hidden_state_size, embedding_size])
        # combine local hidden state and aggregated embedding
        self.mlp2 = Seq(
            *[MLP([embedding_size + node_feature_size, hidden_state_size]), Lin(hidden_state_size, hidden_state_size)])
        # ============================== MPGNN LAYER 2 ==========================
        # local embedding layers (for the hidden states + edge feature)
        self.conv2 = th.nn.Conv2d(2, 2, 6)
        # self.mlp3 = MLP([2*((K-6)+1)*((N*2-6)+1) + hidden_state_size, hidden_state_size, embedding_size])
        self.mlp3 = Seq(*[MLP([2 * ((K - 6) + 1) * ((N * 2 - 6) + 1) + hidden_state_size, hidden_state_size]),
                          Lin(hidden_state_size, embedding_size)])
        # combine local hidden state and aggregated embedding
        self.mlp4 = MLP([embedding_size + node_feature_size + hidden_state_size, hidden_state_size, hidden_state_size])
        # ============================== MPGNN LAYER 3 ==========================
        self.conv3 = th.nn.Conv2d(2, 2, 6)
        # local embedding layers (for the hidden states + edge feature)
        # self.mlp5 = MLP([2*((K-6)+1)*((N*2-6)+1) + hidden_state_size, hidden_state_size, embedding_size])
        self.mlp5 = Seq(*[MLP([2 * ((K - 6) + 1) * ((N * 2 - 6) + 1) + hidden_state_size, hidden_state_size]),
                          Lin(hidden_state_size, embedding_size)])
        # combine local hidden state and aggregated embedding
        self.mlp6 = MLP([embedding_size + node_feature_size + hidden_state_size, hidden_state_size, hidden_state_size])
        # ============================== MPGNN LAYER 4 ==========================
        # local embedding layers (for the hidden states + edge feature)
        self.conv4 = th.nn.Conv2d(2, 2, 6)
        # self.mlp7 = MLP([2*((K-6)+1)*((N*2-6)+1)+ hidden_state_size, hidden_state_size, embedding_size])
        self.mlp7 = Seq(*[MLP([2 * ((K - 6) + 1) * ((N * 2 - 6) + 1) + hidden_state_size, hidden_state_size]),
                          Lin(hidden_state_size, embedding_size)])
        # combine local hidden state and aggregated embedding
        self.mlp8 = Seq(
            *[MLP([embedding_size + node_feature_size + hidden_state_size, hidden_state_size]),
              Lin(hidden_state_size, hidden_state_size)])

        self.gnn1 = Auto_IGConv(self.conv1, self.mlp1, self.mlp2, args.num_agent, args.edge_idx, 1, args)
        self.gnn2 = Auto_IGConv(self.conv2, self.mlp3, self.mlp4, args.num_agent, args.edge_idx, 2, args)
        self.gnn3 = Auto_IGConv(self.conv3, self.mlp5, self.mlp6, args.num_agent, args.edge_idx, 3, args)
        self.gnn4 = Auto_IGConv(self.conv4, self.mlp7, self.mlp8, args.num_agent, args.edge_idx, 4, args)
        # self.conv5 = Auto_IGConv(self.mlp9, self.mlp10, args.num_agent, args.edge_idx, 5)

        # self.fc = nn.Linear(hidden_state_size+node_feature_size, total_dim_peragent)  # size of output
        self.fc = Seq(*[MLP([node_feature_size + hidden_state_size, hidden_state_size]),
                        Lin(hidden_state_size, total_dim_peragent + args.num_user)])  # size of output (K-dimension for tx power allocation)

        self.arch_parameters = nn.Parameter(1e-3 * th.randn(args.num_layer-1, (1 + embedding_size)*2))


    def set_tau(self, tau):
        self.tau = tau

    def get_tau(self):
        return self.tau



    def forward(self, inputs,train_mode=True):
        """
         # inputs: a graph, {"x": node_feature, "edge_atrr": edge_feature}
         # train_mode: True during training or validation
        """
        def get_gumbel_prob(x_input, train_mode=True):
            if train_mode:
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
            else:
                logits = (x_input.log_softmax(dim=-1)) / 0.1
                probs = nn.functional.softmax(logits, dim=-1)  # probabilities for sampling
                """======== for inference ======="""
                index = probs.max(-1, keepdim=True)[1]  # argmax of the final dim
                one_h = th.zeros_like(logits).scatter_(-1, index, 1.0)  # one-hot vector
                """======== for back-propogation/training ======="""
                hardwts = one_h - probs.detach() + probs
            return hardwts, index

        # layer gating [num_layer-1, 2]
        alpha_M, alpha_M_argmax = get_gumbel_prob(self.arch_parameters[:, :2].view(self.args.num_layer-1, 2), train_mode)
        # embedding gating [num_layer-1, embedding_size, 2]
        alpha_E, alpha_E_argmax = get_gumbel_prob(self.arch_parameters[:, 2:].view(self.args.num_layer-1, self.args.embedding_size, 2),train_mode)
        self.alpha_M, self.alpha_E = alpha_M.detach(), alpha_E.detach()

        x0, edge_attr = inputs['x'], inputs['edge_attr']
        batch_size = x0.shape[0]


        x1 = self.gnn1(x0, x0, edge_attr,
                            alpha_E=th.tensor([1, 0], device=x0.device).repeat((self.args.embedding_size,1)),
                            alpha_M=th.tensor([1, 0], device=x0.device),
                            dynamic=False)
        x2 = self.gnn2(x0, x1, edge_attr, alpha_E[0].view(self.args.embedding_size,-1), alpha_M[0])
        x3 = self.gnn3(x0, x2, edge_attr, alpha_E[1].view(self.args.embedding_size,-1), alpha_M[1])
        x4 = self.gnn4(x0, x3, edge_attr, alpha_E[2].view(self.args.embedding_size, -1), alpha_M[2])

        y = self.fc(th.cat([x0.view(batch_size * self.args.num_agent, -1),
                x4.view(batch_size * self.args.num_agent, -1)], dim=-1))
        N,K = self.args.num_antenna, self.args.num_user

        power_user = 0.01 + (1-0.01*self.args.num_user)*th.softmax((y[:,:K]).view(y.shape[0],-1), dim=-1)
        BF = th.tanh(y[:, K:(N * K * 2 + K)])

        SIC, _ = get_gumbel_prob(y[:, (K + N * K * 2):].view(y.shape[0], int(K * (K - 1) / 2), 3))
        out = th.cat((power_user, BF, SIC.view(y.shape[0], -1)), dim=-1)  # [batch_size*num_agent, 2*N*K+K*(K-1)/2*3]

        return out, alpha_M, alpha_E



    def get_weights(self):
        """
        We empirically found that partially train the following GNN parameters achieve better results
        """
        xlist = list(self.conv1.parameters()) + list(self.conv2.parameters())
        xlist += list(self.mlp1.parameters()) + list(self.mlp2.parameters()) + list(self.mlp3.parameters()) + list(
            self.mlp4.parameters())
        xlist += list(self.conv3.parameters()) + list(self.conv4.parameters())
        # xlist += list(self.mlp5.parameters()) + list(self.mlp6.parameters()) + list(self.mlp7.parameters()) + list(self.mlp8.parameters())
        xlist += list(self.fc.parameters())
        return xlist

    def get_weights_dict(self):
        x = {}
        x["mlp1"] = list(self.mlp1.parameters())
        x["mlp2"] = list(self.mlp2.parameters())
        x["mlp3"] = list(self.mlp3.parameters())
        x["mlp4"] = list(self.mlp4.parameters())
        x["mlp5"] = list(self.mlp5.parameters())
        x["mlp6"] = list(self.mlp6.parameters())
        x["mlp7"] = list(self.mlp7.parameters())
        x["mlp8"] = list(self.mlp8.parameters())
        x["fc"] = list(self.fc.parameters())
        return x

    def get_alphas(self):
        return [self.arch_parameters]


    def get_comm_cost(self):
        return ((self.alpha_M[:,0].unsqueeze(dim=-1)) * self.alpha_E[:,:,0]).sum() + self.args.embedding_size


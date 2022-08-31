
import torch
import numpy as np

"""
============================ build environment =========================
"""


def BuildEnv(args):
    args.save_dir += 'M_' + str(args.num_agent) + '_K_' + str(args.num_user) + '_intf_corr_' + str(
        args.intf_corr) + '_data_corr' + str(args.data_corr) + '_layer_' + str(args.num_layer) + '_emb_' +str(
        args.embedding_size)  +'/'

    args.NNoutput_size = int(2 * args.num_antenna * args.num_user + 3 * 0.5 * args.num_user * (args.num_user - 1))
    args.sigma = 1 # sigma
    args.Pmax = 10 ** (args.SNR / 10) # Maximum transmission power at each BS, calculate according to SNR [W]
    args.Rmin = 0.3
    args.penalty_rate = 2
    M = args.num_agent
    args.edge_idx = torch.zeros([int(M * (M - 1)), 2]) # (Source_node, End_node) of each edge
    args.corr = torch.eye(M) * args.data_corr + (torch.ones(M) - torch.eye(M)) * args.intf_corr
    i = 0
    for n in range(M):
        for m in range(M):
            if not n==m:
                args.edge_idx[i, :] = torch.tensor([n, m]) # [tx_node, rx_node]
                i += 1
    args.SIC_src = get_SIC_scatter_src(args)
    args.node_feature_size = 2 * args.num_user * args.num_antenna
    args.edge_feature_size = 2 * args.num_user * args.num_antenna
    args.edge_idx = args.edge_idx.long()
    return args

def get_SIC_scatter_src(args):
    K = args.num_user
    c = torch.linspace(1,K**2,K**2).view(K,K)
    d = torch.zeros(K*(K-1))
    n = 0
    for i in range(K):
        for j in range(K):
            if i < j:
                d[n] = c[i,j]
                d[n+1] = c[j,i]
                n += 2
    d = d - 1
    return d



"""
=================== generate graph data for training/test==================
"""


def buid_graph(H, args):
    """H: [bs, M', M, K, N] 基站m'对用户(m,k)的信道"""
    bs, M, N, K = args.batch_size, args.num_agent, args.num_antenna, args.num_user
    data = {'H': H}
    x = torch.zeros([bs, M, N * K * 2])  # node feature
    edge_attr = torch.zeros([bs, len(args.edge_idx), N * K * 2])  # edge_attr[sample_id, edge_id, H[:,tx_node, serving_node, k, :]]
    i = 0
    for n in range(M):
        for m in range(M):
            if m == n:
                x[:, m, :] = torch.cat(
                    (H[:, m, m, :, :].real.unsqueeze(dim=-1), H[:, m, m, :, :].imag.unsqueeze(dim=-1)), dim=-1).view(bs, -1)
            else:
                ee = torch.cat((H[:, n, m, :, :].real.unsqueeze(dim=-1), H[:, n, m, :, :].imag.unsqueeze(dim=-1)), dim=-1).view(bs, -1)
                edge_attr[:, i, :] = ee
                i += 1
    data["x"] = x.cuda()
    data["edge_attr"] = edge_attr.cuda()
    return data

def build_CNN_inputs(H, args):
    """
    H, [bs, M', M, K, N]----- the channel from BS m' to user (m,k)
    """
    data = {'H': H}
    bs, M, N_tx, K = args.batch_size, args.num_agent, args.num_antenna, args.num_user
    x = torch.zeros([bs, M*M, N_tx, K * 2])  # compat channel matrix
    H = H.permute([0,1,2,4,3])
    i = 0
    for m in range(M):
        for n in range(M):
            x[:,i] = torch.cat((H[:, m, n, :, :].real.unsqueeze(dim=-1), H[:, m, n, :, :].imag.unsqueeze(dim=-1)), dim=-1).view(bs,N_tx,K*2)
            i += 1
    data["x"] = x.view(bs,M*M*N_tx,K*2).cuda()
    return data

def build_MLPs_inputs(H, args):
    """
    H, [bs, M', M, K, N]----- the channel from BS m' to user (m,k)
    """
    data = {'H': H}
    bs, M, N_tx, K = args.batch_size, args.num_agent, args.num_antenna, args.num_user
    x = torch.zeros([bs, M*M,N_tx*K*2])  # compat channel matrix
    i = 0
    for m in range(M):
        for n in range(M):
            x[:,i] = torch.cat((H[:, m, n, :, :].real.unsqueeze(dim=-1), H[:, m, n, :, :].imag.unsqueeze(dim=-1)), dim=-1).view(bs,N_tx*K*2)
            i += 1
    data["x"] = x.view(bs,M*M*N_tx*K*2).cuda()
    return data
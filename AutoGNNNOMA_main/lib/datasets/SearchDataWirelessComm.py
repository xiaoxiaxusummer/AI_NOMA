import torch, random
import torch.utils.data as data
import math
from train_func import calEleNorm2
import scipy.io as scio
import os


class DatasetNOMA(data.Dataset):
    def __init__(self, name, args, train=True, saveData=False, small_sample=False):
        self.datasetname = name
        self.train = train
        self.args = args

        bs = args.batch_size

        if train:
            """" ====================== train & test dataset ======================="""
            self.buffer_size = 50 # Number of sample batches in the buffer, ratio 4:1 for training and test, respectively
            self.train_buffer= torch.linspace(1, bs*40, bs*40).long()-1
            self.test_buffer = torch.linspace(bs*40+1,bs*self.buffer_size,bs*50).long()-1
            self.length = len(self.train_buffer)
        else:
            """" ==================== validation dataset ===================="""
            if small_sample:
                self.buffer_size = 5  # Number of sample batches in the buffer, used for compare learning & conventional optimization alg.
            else:
                self.buffer_size = 10 # Number of sample batches in the buffer, used for compare different learning alg.
            self.train_buffer = []
            self.test_buffer = torch.linspace(1,bs*self.buffer_size,bs*self.buffer_size).long()-1
            self.length = len(self.test_buffer)

        self.data = []
        self.renew_data()
        if saveData:
            file = './env_data/M_'+str(args.num_agent)+'_K_'+str(args.num_user)+'_intf_corr_'+str(args.intf_corr)+'_data_corr_'\
                   + str(args.data_corr)+'_buffer_' + str(self.buffer_size) + '.mat'
            if os.path.isfile(file):
                data = scio.loadmat(file)
                self.data = torch.tensor(data["H"])
            else:
                scio.savemat(file, {'H': self.data.resolve_conj().numpy()})

    def __repr__(self):
        return ('{name}(name={datasetname}, train={tr_L}, valid={val_L})'.format(name=self.__class__.__name__,
                                                                                 datasetname=self.datasetname,
                                                                                 tr_L=len(self.train_buffer),
                                                                                 val_L=len(self.test_buffer)))

    def __len__(self):
        if self.train:
            return len(self.train_buffer)
        else:
            return len(self.test_buffer)


    def __getitem__(self, index):
        if self.train:
            train_index = self.train_buffer[index]
            valid_index = random.choice(self.test_buffer)
            train_data = self.data[train_index]
            valid_data = self.data[valid_index]
            return train_data, valid_data
        else:
            valid_index = self.test_buffer[index]
            valid_data = self.data[valid_index]
            return valid_data



    """
    =================== generate samples for training/test==================
    """
    def renew_data(self):
        """H: [bs, M', M, K, N] 基站m'对用户(m,k)的信道"""
        args = self.args
        n_batch, bs, M, N, K = self.buffer_size, args.batch_size, args.num_agent, args.num_antenna, args.num_user
        H = CalChannel(n_batch * bs, M, N, K, args.corr)
        self.data = H # [n_batch*bs, M(tx BSs), M'(serving BSs), K, N] ========= H[:, m', m, k, :] 基站m'对用户(m,k)的信道


"""
=================== generate channel for the given correlation ==================
"""


def CalChannel(bs, M, N, K, corr_all):
    dist_user = torch.zeros([bs, M, M, K])
    d0 = 50
    dr = 200
    pathloss_exp = 3
    for m in range(M):
        dist_user[:, m, m, :] = d0 + torch.rand([bs, K]) * (dr-d0) # data channels
        dist_user[:, m, :m, :] = 100 + torch.rand([bs, m, K]) * 200 # interference channels
        dist_user[:, m, (m+1):, :] = 100 + torch.rand([bs, M - m - 1, K]) * 200 # interference channels
    pathloss = (torch.div(1,(1+(dist_user/d0)**pathloss_exp))).sqrt()

    H_ori = 1 / math.sqrt(2) * torch.complex(torch.randn([bs, M, M, N, K]), torch.randn([bs, M, M, N, K]))
    H_all = torch.zeros([bs, M, M, N, K]) * 0j
    for m in range(M):
        for n in range(M):
            if m == n:
                sortH = True
            else:
                sortH = False
            H_all[:, m, n, :, :] = GenChannelWithCorr(N, K, H_ori[:, m, n, :, :], pathloss[:, m, n, :], corr_all[m, n],sortH)

    H_all = H_all.permute([0, 1, 2, 4, 3]).conj() # [bs, M, M', K, N] 基站m对用户(m',k)的信道

    return H_all


def GenChannelWithCorr(N, K, H_ori, pathloss, corr, sortH = False):
    """
    H_ori: [bs,N,K]
    pathloss: [bs,K]
    """
    bs = H_ori.shape[0]
    vecnorm_H = torch.sum(calEleNorm2(H_ori), dim=-2).view([bs,1,K])
    H_normalized_t = torch.zeros([bs,N,K])*0j
    for k in range(K):
        H_normalized_t[:,:,k] = torch.div(H_ori[:,:,k], calEleNorm2(H_ori[:,:,k]).sum(dim=-1).view(bs,1))
    H_normalized = torch.div(H_ori, vecnorm_H)
    R = (torch.eye(K)*(1+0j) / 2).repeat([bs, 1, 1]) # [BS, K, K]
    phi0 = torch.rand([bs, 1]) * 2 * math.pi - 1
    for i in range(K):
        for j in range(i + 1, K):
            phi = phi0 + torch.rand([bs, 1]) * (1 - corr)
            R[:, i, j] = ((corr * torch.exp(phi*1j)) ** (j - i)).view([bs])
    R = R + R.permute([0, 2, 1])
    H_normalized = torch.matmul(H_normalized, (R ** (1 / 2)))
    H = H_normalized * vecnorm_H  # [N,bs,K]
    H = H * pathloss.view([bs,1,K])  # [bs,N,K]
    if sortH:
        idx = torch.argsort(torch.sum(calEleNorm2(H), dim=-2))
        H = H.gather(dim=-1, index=idx.repeat([1,1,N]).view([bs,N,K]))
    return H

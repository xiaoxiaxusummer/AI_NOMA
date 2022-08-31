import torch as th
import torch.nn as nn
import numpy as np
from copy import deepcopy
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
import math

th.autograd.set_detect_anomaly(True)


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())  # , BN(channels[i]))
        for i in range(1, len(channels))
    ])


class MLPs(nn.Module):

    def __init__(self, args):
        super(MLPs, self).__init__()

        self.M = args.num_agent
        self.K = args.num_user
        self.N = args.num_antenna
        self.args = args
        self.output_NN_size = (int(args.NNoutput_size) + self.K) * self.M
        self.in_channels = 1
        self.input_size = self.M*self.M*self.N*self.K*2

        self.cfgs =  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        self.mlp1 = MLP([self.input_size,128,64])
        self.mlp2 = MLP([64,64,64])
        self.fc = nn.Sequential(  # Fully connected layers
            nn.Linear(64,64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.output_NN_size)
        )



    def forward(self, inputs, train_mode=True):
        """
            # inputs: a mat, size = (M*M*N, K*2)
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

        x = inputs["x"]
        bs = x.shape[0]
        y1 = self.mlp1(x)
        y2 = self.mlp2(y1)
        y = self.fc(y2).view(bs, self.M, -1)

        N, K = self.N, self.K
        power_user = 0.01 + (1 - 0.01 * self.K) * th.softmax((y[:, :, :K]).view(y.shape[0], self.M, -1), dim=-1)
        BF = th.tanh(y[:, :, K:(N * K * 2 + K)])

        SIC, _ = get_gumbel_prob(y[:, :, (K + N * K * 2):].view(y.shape[0], self.M, int(K * (K - 1) / 2), 3))

        out = th.cat((power_user, BF, SIC.view(y.shape[0], self.M, -1)), dim=-1)  # [batch_size*num_agent, 2*N*K+K*(K-1)/2*3]
        return out

    def set_tau(self, tau):
        self.tau = tau

    def get_tau(self):
        return self.tau


    def get_comm_cost(self):
        return self.M* (self.M*self.N*self.K*2 + self.K*(self.K-1) +  self.N*self.K*2)

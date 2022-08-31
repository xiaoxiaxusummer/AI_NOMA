import torch as th
import torch.nn as nn
import numpy as np
from copy import deepcopy
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
import math

th.autograd.set_detect_anomaly(True)



def make_layer(cfgs):
    layers = []
    in_channel = 1
    for cfg in cfgs:
        if cfg == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channel, out_channels=cfg, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channel = cfg
    return nn.Sequential(*layers)

class ConvNet(nn.Module):

    def __init__(self, args):
        super(ConvNet, self).__init__()

        self.M = args.num_agent
        self.K = args.num_user
        self.N = args.num_antenna
        self.args = args
        self.output_NN_size = (int(args.NNoutput_size) + self.K) * self.M
        self.in_channels = 1

        self.cfgs =  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        self.features = make_layer(self.cfgs)  # Conv layers
        self.fc = nn.Sequential(  # Fully connected layers
            nn.Linear(in_features=512 * (self.M*self.M*self.N-5) * (self.K*2-5) , out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, self.output_NN_size)
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
        feature = self.features(x.unsqueeze(dim=1))
        linear_input = th.flatten(feature, 1)
        y = self.fc(linear_input).view(bs, self.M, -1)

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

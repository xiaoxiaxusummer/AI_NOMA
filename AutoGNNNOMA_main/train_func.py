import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path
from lib.log_utils import AverageMeter, time_string, convert_secs2time
from lib.utils.graph_utils import *

from pathlib import Path
from lib.procedures import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler


lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from lib.config_utils import load_config, dict2config, configure2str
# from lib.datasets import get_datasets, SearchDataset
# from lib.procedures import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from lib.log_utils import AverageMeter, time_string, convert_secs2time


"""
==================== return loss for learning network parameters ======================
"""
def calEleNorm2(X):
    return torch.square(torch.real(X)) + torch.square(torch.imag(X))


def sr_loss(inputs, out, alpha_M, alpha_E, args):
    bs, M, K, N_tx = args.batch_size, args.num_agent, args.num_user, args.num_antenna
    out = out.view(bs, M, -1)  # [batch_size, M, 1+N*K*2 + K*(K-1)*3]
    device = out.device
    SIC_src = args.SIC_src.to(device).long()

    BF_power = (args.Pmax*out[:, :, :K].view([bs,M,K]).repeat_interleave(N_tx,dim=-2).view(bs,M,N_tx,K)).sqrt() # [bs,M,1,K]
    BF_ori = out[:, :, K:(K+K*N_tx)].view(bs, M, N_tx, K) + out[:, :, (K + K * N_tx): (K + K * N_tx * 2)].view(bs, M, N_tx, K) * 1j  # [bs,M,N,K]
    norm_BF = calEleNorm2(BF_ori).sum(dim=-2).repeat_interleave(N_tx,dim=-2).view(bs,M,N_tx,K)
    BF = torch.div(BF_ori, norm_BF.sqrt())*BF_power
    # BF = torch.mul(BF_ori, torch.sqrt(torch.div(args.Pmax, norm2_BF)*BF_power).unsqueeze(dim=-1).unsqueeze(dim=-1))

    # SIC operation
    beta = torch.eye(K, device=device).repeat(bs*M,1).view(bs,M,K*K)
    beta_vec = out[:, :, (K + K * N_tx * 2):].view([bs, M, int(K * (K - 1) / 2), 3])
    beta = beta.scatter(-1, SIC_src.repeat(bs*M,1).view(bs,M,K*(K-1)) ,beta_vec[:,:,:,:2].reshape(bs,M,K*(K-1))).view(bs,M,K,K)

    H = inputs["H"].cuda()  # [bs, M', M, K, N] (channel from BS m'to user (m,k))
    H_permute = H.permute([0, 2, 1, 3, 4])  # [bs, M, M', K, N] (channel from BS m'to user (m,k))

    G = calEleNorm2((torch.matmul(H_permute.reshape(bs * M, M, K, N_tx), BF.repeat_interleave(M, dim=0))))\
                .view(bs, M, M, K, K)  # [bs, M, M', K, K'] signal gain of user (m',k‘) received at user (m,k)

    I_M = torch.eye(M, device=device)  # [M,M]

    ICI_gain = torch.sum(torch.mul((1 - I_M).expand([bs, K, K, M, M]), G.permute([0, 3, 4, 1, 2])),
                         dim=-1)  # [bs, K, K',M] the inter-cell interference gain suffered by user (m,k)
    ICI = (torch.sum(ICI_gain, dim=-2)).permute([0, 2, 1]).repeat_interleave(K, dim=-1).view(
        [bs, M, K, K]) + args.sigma   # [bs, M, K] the ICI suffered by user (m,k) + AWGN


    # effective gain for user (m,k) for decoding the signal of user (m,k')
    decoding_gain = (I_M.expand(bs,M,M).unsqueeze(-1).unsqueeze(-1)*G).sum(dim=1) # [bs, M, K, K']

    # alpha_{iu} * alpha_{uk}  --- [bs,M,I,K,U]
    beta_iu_uk = beta.repeat_interleave(K, dim=2).view([bs, M, K, K, K]) * beta.permute([0, 1, 3, 2]).repeat_interleave(
        K, dim=1).view([bs, M, K, K, K])
    # [bs,M,I,K,U] extract/exclude i==k
    E = (torch.eye(K, device=device).expand([bs, M, K, K, K]).permute([0, 1, 3, 4, 2]))
    # [bs,M,I,K,U] for k > u, whether the interference from (m,u) exists when user (m,i) decoding signal of (m,k)
    beta_tril = 1 - beta.repeat_interleave(K, dim=2).view([bs, M, K, K, K]) + beta_iu_uk  # [bs,M,I,K,U] for u < k, whether interference exists when user i decode the signal of k
    #  [bs,M,I,K,U] for k < u, whether the interference from (m,u) exists when user (m,i) decoding signal of (m,k)
    beta_triu = 1 - beta.repeat_interleave(K, dim=2).view([bs, M, K, K, K]) * beta.repeat_interleave(K, dim=1).view(
        [bs, M, K, K, K])

    #  [bs,M,I,K,U]
    triu_mask = (1 - E) * (torch.ones([K, K], device=device).triu() - torch.eye(K, device=device)).expand(
        [bs, M, K, K, K])
    tril_mask = (1 - E) * (torch.ones([K, K], device=device).tril() - torch.eye(K, device=device)).expand(
        [bs, M, K, K, K])

    #  [bs,M,I,K,U] whether the interference from u exists when user i decode the signal of k, i \ne k
    mask = triu_mask * beta_triu + tril_mask * beta_tril

    # [bs,M,I,K,U] whether the interference from u exists when user i decode the signal of k, i == k
    mask_comm = E * ((1 - beta).repeat_interleave(K, dim=1).view([bs, M, K, K, K]))

    # [bs,M,K,I,U] -> [bs,M,I,K] the interference for user i to decode the signal of k
    decoding_intf = torch.sum(torch.mul(decoding_gain.repeat_interleave(K, dim=1).view([bs, M, K, K, K]),
                                        mask.permute([0, 1, 3, 2, 4]) + mask_comm.permute([0, 1, 3, 2, 4])),
                              dim=-1).permute([0, 1, 3, 2])

    decoding_rate = torch.log2(1 + torch.div(decoding_gain, decoding_intf + ICI))

    comm_rate = (decoding_rate * (torch.eye(K, device=device).repeat(bs, M, 1, 1).view(bs, M, K, K))).sum(dim=-1)  # [bs,M,K]  communication rate
    d_rate = decoding_rate * beta + (comm_rate.repeat_interleave(K, dim=-2).view(bs, M, K, K)) * (1 - beta)

    rate, _ = torch.min(d_rate, dim=-2) # [bs,M,K]
    vio_rate = torch.nn.functional.relu(args.Rmin - rate) # relu(x) = max(x,0)
    sum_rate = rate.mean(dim=0).sum()  # scalar
    loss = torch.neg(sum_rate) +  (vio_rate).mean(dim=0).sum() * args.penalty_rate

    return loss, rate.detach().mean(dim=0).sum(), beta.detach().mean(dim=0).sum() - (M * K)


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


def _hessian_vector_product(vector, network, args, base_inputs, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(network.get_weights(), vector):
        p.data.add_(v,alpha=R)
    out, alpha_M, alpha_E = network(base_inputs)
    loss, _, _ = sr_loss(base_inputs, out, alpha_M, alpha_E, args)
    grads_p = torch.autograd.grad(loss, network.get_alphas())

    for p, v in zip(network.get_weights(), vector):
        p.data.sub_(v, alpha = 2 * R)
    out, _, _ = network(base_inputs)
    loss, _, _ = sr_loss(base_inputs, out, alpha_M, alpha_E, args)
    grads_n = torch.autograd.grad(loss, network.get_alphas())

    for p, v in zip(network.get_weights(), vector):
        p.data.add_(v,alpha=R)
    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]


def backward_step_unrolled(network, args, base_inputs ,w_optimizer, arch_inputs):
    LR, WD, momentum = w_optimizer.param_groups[0]['lr'], w_optimizer.param_groups[0]['weight_decay'], \
                       w_optimizer.param_groups[0]['momentum']
    # _compute_unrolled_model
    NN_output, alpha_M, alpha_E = network(base_inputs)
    loss, _, _ = sr_loss(base_inputs, NN_output, alpha_M, alpha_E, args)
    with torch.no_grad():
        theta = _concat(network.get_weights())
        try:
            moment = _concat(w_optimizer.state[v]['momentum_buffer'] for v in network.get_weights())
            moment = moment.mul_(momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, network.get_weights())) + WD * theta
        params = theta.sub_(moment + dtheta, alpha=LR)
    unrolled_model = deepcopy(network)
    model_dict = unrolled_model.state_dict()
    new_params, offset = {}, 0
    for k, v in network.named_parameters():
        if 'arch_parameters' in k: continue
        if ('mlp5' in k) or ('mlp6' in k) or ('mlp7' in k) or ('mlp8' in k): continue
        v_length = np.prod(v.size())
        new_params[k] = params[offset: offset + v_length].view(v.size())
        offset += v_length
    model_dict.update(new_params)
    unrolled_model.load_state_dict(model_dict)

    unrolled_model.zero_grad()
    unrolled_outputs, unrolled_alpha_M, unrolled_alpha_E = unrolled_model(arch_inputs)
    unrolled_loss, unrolled_rate, unrolled_SIC = sr_loss(arch_inputs, unrolled_outputs, unrolled_alpha_M,
                                                         unrolled_alpha_E, args)
    unrolled_loss.backward()

    dalpha = unrolled_model.arch_parameters.grad
    vector = [v.grad.data for v in unrolled_model.get_weights()]
    [implicit_grads] = _hessian_vector_product(vector, network, args, base_inputs)

    dalpha.data.sub_(implicit_grads.data, alpha=LR)

    if network.arch_parameters.grad is None:
        network.arch_parameters.grad = deepcopy(dalpha)
    else:
        network.arch_parameters.grad.data.copy_(dalpha.data)

    comm_cost = (unrolled_alpha_M[:, 0].unsqueeze(dim=-1) * unrolled_alpha_E[:, :, 0]).sum()
    return unrolled_loss.detach(), unrolled_rate.detach(), unrolled_SIC.detach(), comm_cost.detach()


def search_func(xloader, network, args, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, logger):
    data_time, batch_time = AverageMeter(), AverageMeter()
    arch_losses, arch_rates, arch_SICs, arch_comm_costs, h_betas, h_archs = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    network.train()
    end = time.time()

    for step, (base_inputs, arch_inputs) in enumerate(xloader):
        # build graph model
        base_inputs_graph, arch_inputs_graph = buid_graph(base_inputs, args), buid_graph(arch_inputs, args)
        scheduler.update(None, 1.0 * step / len(xloader))

        # measure data loading time
        data_time.update(time.time() - end)
        # update the architecture-weight
        a_optimizer.zero_grad()
        arch_loss, arch_rate, arch_SIC, arch_comm_cost = backward_step_unrolled(network, args, base_inputs_graph, w_optimizer,
                                                        arch_inputs_graph)
        a_optimizer.step()

        # update the weights
        w_optimizer.zero_grad()
        NN_output, alpha_M, alpha_E = network(base_inputs_graph)
        # calculate loss
        base_loss, _, _ = sr_loss(base_inputs_graph, NN_output, alpha_M, alpha_E, args)
        base_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        w_optimizer.step()

        # record
        arch_losses.update(arch_loss.item(), len(arch_inputs))
        arch_rates.update(arch_rate.item(), len(arch_inputs))
        arch_SICs.update(arch_SIC.item(), len(arch_inputs))
        arch_comm_costs.update(arch_comm_cost.item(), len(arch_inputs))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0 or step + 1 == len(xloader):
            Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
            Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(
                batch_time=batch_time, data_time=data_time)
            Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  rate {rate.val:.2f} ({rate.avg:.2f}) SIC {SIC.val:.2f} ({SIC.avg:.2f}) cost {cost.val:.2f} ({cost.avg:.2f})]'.format(
                loss=arch_losses, rate=arch_rates, SIC=arch_SICs, cost=arch_comm_costs)
            logger.log(Sstr + ' ' + Tstr + ' ' + Astr)

            # comm_cost = network.module.get_comm_cost()
            comm_cost = network.get_comm_cost()
            logger.log(comm_cost)
    return arch_losses, arch_rates.avg, arch_comm_costs.avg


def valid_func(xloader, network, args):
    data_time, batch_time = AverageMeter(), AverageMeter()
    arch_losses, arch_rates, arch_SICs, arch_comm_costs = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    network.eval()
    end = time.time()
    with torch.no_grad():
        for step, (arch_inputs) in enumerate(xloader):
            # measure data loading time
            data_time.update(time.time() - end)
            # prediction
            arch_input_graph = buid_graph(arch_inputs, args)
            outputs, alpha_M, alpha_E = network(arch_input_graph, train_mode=False)
            arch_loss, arch_rate, arch_SIC = sr_loss(arch_input_graph, outputs, alpha_M, alpha_E, args)
            arch_comm_cost = (alpha_M[:, 0].unsqueeze(dim=-1) * alpha_E[:, :, 0]).sum()
            # record
            arch_losses.update(arch_loss.item(), arch_inputs.size(0))
            arch_rates.update(arch_rate.item(), arch_inputs.size(0))
            arch_SICs.update(arch_SIC.item(), arch_inputs.size(0))
            arch_comm_costs.update(arch_comm_cost.item(), arch_inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    return arch_losses.avg, arch_rates.avg, arch_SICs.avg, arch_comm_costs.avg



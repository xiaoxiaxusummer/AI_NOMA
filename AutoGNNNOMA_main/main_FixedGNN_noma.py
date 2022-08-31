from train_func import *
from algs.fixedgnn_conv import FixedMPGNN
import torch
from lib.datasets.SearchDataWirelessComm import DatasetNOMA

def get_comm_cost(args):
    return torch.sum(torch.sum((args.alpha_M[:, 0].t()) * (args.alpha_E[:, :, 0].t()), dim=0))  + args.embedding_size

def train_fixed_model(xloader, network, args, scheduler, w_optimizer, epoch_str, print_freq, logger):
    data_time, batch_time = AverageMeter(), AverageMeter()
    losses, rates, SICs, comm_costs = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    network.train()
    end = time.time()

    for step, (inputs, _) in enumerate(xloader):
        inputs_graph = buid_graph(inputs, args)
        scheduler.update(None, 1.0 * step / len(xloader))
        data_time.update(time.time() - end)  # measure data loading time
        w_optimizer.zero_grad()
        NN_output = network(inputs_graph)
        # calculate loss
        loss, rate, SIC = sr_loss(inputs_graph, NN_output, args.alpha_M, args.alpha_E, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.get_weights(), 5)
        w_optimizer.step()
        comm_cost = network.get_comm_cost()

        # record
        losses.update(loss.item(), len(inputs))
        rates.update(rate.item(), len(inputs))
        SICs.update(SIC.item(), len(inputs))
        comm_costs.update(comm_cost.item(), len(inputs))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0 or step + 1 == len(xloader):
            Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
            Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(
                batch_time=batch_time, data_time=data_time)
            Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  rate {rate.val:.2f} ({rate.avg:.2f}) SIC {SIC.val:.2f} ({SIC.avg:.2f}) cost {cost.val:.2f} ({cost.avg:.2f})]'.format(
                loss=losses, rate=rates, SIC = SICs, cost=comm_costs)
            logger.log(Sstr + ' ' + Tstr + ' ' + Astr)

            comm_cost = get_comm_cost(xargs)
            logger.log(comm_cost)
    return losses, rates.avg, SICs.avg, comm_costs.avg

def test_fixed_model(xloader, network, args):
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
            NN_output = network(arch_input_graph)
            arch_loss, arch_rate, arch_SIC = sr_loss(arch_input_graph, NN_output, args.alpha_M, args.alpha_E, args)
            arch_comm_cost = network.get_comm_cost()
            # record
            arch_losses.update(arch_loss.item(), arch_inputs.size(0))
            arch_SICs.update(arch_SIC.item(), arch_inputs.size(0))
            arch_rates.update(arch_rate.item(), arch_inputs.size(0))
            arch_comm_costs.update(arch_comm_cost.item(), arch_inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    return arch_losses.avg, arch_rates.avg, arch_SICs.avg, arch_comm_costs.avg


def main_train(xargs):
    xargs = BuildEnv(xargs)
    xargs.alpha_M = torch.cat((torch.ones([xargs.num_layer-1,1]),torch.zeros([xargs.num_layer-1,1])),dim=-1)
    xargs.alpha_E = torch.cat((torch.ones([xargs.num_layer-1,xargs.embedding_size,1]),
                               torch.zeros([xargs.num_layer-1,xargs.embedding_size,1])),dim=-1)

    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(xargs)
    fixedmodel = FixedMPGNN(xargs)
    logger.log('search-model :\n{:}'.format(fixedmodel))

    config_path = 'configs/FixedGNN.config'
    config = load_config(config_path, {'num_agent': xargs.num_agent, 'num_antenna': xargs.num_antenna, 'optim':'SGD'}, logger)
    xargs.batch_size = config.batch_size

    # optimizers of weights
    w_optimizer, w_scheduler = get_optim_scheduler(fixedmodel.get_weights(), config)

    logger.log('w-optimizer : {:}'.format(w_optimizer))
    logger.log('w-scheduler : {:}'.format(w_scheduler))

    # network = torch.nn.DataParallel(search_model,device_ids=[0]).cuda()
    network = fixedmodel.cuda()

    """
    ====================================  automatically resume from previous checkpoint  ====================================
    """
    last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
    if last_info.exists():
        logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
        last_info   = torch.load(last_info)
        start_epoch = last_info['epoch']
        checkpoint  = torch.load(last_info['last_checkpoint'])
        comm_costs   = checkpoint['comm_costs']
        valid_rates = checkpoint['valid_accuracies']
        valid_SICs = {} #checkpoint['valid_SICs']
        fixedmodel.load_state_dict( checkpoint['fixedmodel'] )
        w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
        w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
        logger.log("=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        start_epoch, valid_rates, valid_SICs, comm_costs = 0, {'best': -1}, {}, {}

    """
    ================================================  loading data samples  ================================================
    """
    search_data = DatasetNOMA("cluster-free NOMA", xargs, train=True)
    valid_data = DatasetNOMA("cluster-free NOMA", xargs, train=False, saveData = True, small_sample=True)

    train_loader = torch.utils.data.DataLoader(search_data, batch_size=config.batch_size, shuffle=True,
                                                num_workers=1)#, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, shuffle=False, num_workers=1)#, pin_memory=True)

    logger.log(
        '||||||| Train-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(len(train_loader),
                                                                                                    len(valid_loader),
                                                                                                    config.batch_size))
    logger.log('||||||| Config={:}'.format( config))

    """
    ====================================================  start training  ==================================================
    """
    # start training
    start_time, search_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup
    for epoch in range(start_epoch, total_epoch):
        network.set_tau(
            xargs.tau_max - (xargs.tau_max - xargs.tau_min) * epoch / (total_epoch - 1)
        )
        search_data.renew_data()
        train_loader = torch.utils.data.DataLoader(search_data, batch_size=config.batch_size, shuffle=True,
                                                   num_workers=1)  # , pin_memory=True)
        w_scheduler.update(epoch, 0.0)
        need_time = 'Time Left: {:}'.format(convert_secs2time(epoch_time.val * (total_epoch - epoch), True))
        epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
        min_LR = min(w_scheduler.get_lr())
        logger.log('\n[Search the {:}-th epoch] {:}, LR={:}'.format(epoch_str, need_time, min_LR))
        search_w_loss, search_mean_rate, search_mean_SIC, search_comm_cost = train_fixed_model(train_loader, network, xargs, w_scheduler, w_optimizer, epoch_str, xargs.print_freq, logger)
        search_time.update(time.time() - start_time)
        logger.log('[{:}] training : loss={:.2f}, mean rate={:.2f}, mean SIC={:.2f}, comm. cost={:.2f}, time-cost={:.1f} '.format(
            epoch_str, search_w_loss.avg, search_mean_rate, search_mean_SIC, search_comm_cost, search_time.sum))
        valid_start_time = time.time()
        valid_a_loss, valid_rate, valid_SIC, valid_comm_cost = test_fixed_model(valid_loader, network, xargs)
        valid_time = time.time()-valid_start_time
        logger.log(
            '[{:}] evaluate  : loss={:.2f}, rate={:.2f}, SIC={:.2f}, valid_time={:.2f}, comm. cost={:.2f}'.format(epoch_str, valid_a_loss,
                                                                                           valid_rate, valid_SIC, valid_time, valid_comm_cost))

        valid_rates[epoch] = valid_rate
        valid_SICs[epoch] = valid_SIC
        if valid_rate > valid_rates['best']:
            valid_rates['best'] = valid_rate
            comm_costs['best'] = get_comm_cost(xargs)
            find_best = True
        else:
            find_best = False

        comm_costs[epoch] = get_comm_cost(xargs)
        logger.log('<<<--->>> The {:}-th epoch : {:}'.format(epoch_str, comm_costs[epoch]))
        # save checkpoint
        save_path = save_checkpoint({'epoch': epoch + 1,
                                     'args': deepcopy(xargs),
                                     'fixedmodel': fixedmodel.state_dict(),
                                     'w_optimizer': w_optimizer.state_dict(),
                                     'w_scheduler': w_scheduler.state_dict(),
                                     'comm_costs': comm_costs,
                                     'valid_accuracies': valid_rates,
                                     'valid_SICs': valid_SICs
                                     },
                                    model_base_path, logger)
        last_info = save_checkpoint({
            'epoch': epoch + 1,
            'args': deepcopy(xargs),
            'last_checkpoint': save_path,
        }, logger.path('info'), logger)
        if find_best:
            logger.log('<<<--->>> The {:}-th epoch : find the highest validation rate : {:.2f}%.'.format(epoch_str,
                                                                                                             valid_rate))
            copy_checkpoint(model_base_path, model_best_path, logger)

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()


    logger.log('\n' + '-' * 100)
    # check the performance from the architecture dataset
    logger.log('FixedMPGNN : run {:} epochs, time cost {:.1f} s, last-comm_cost is {:}.'.format(total_epoch, search_time.sum, comm_costs[total_epoch - 1]))
    logger.close()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__=='__main__':
    parser = argparse.ArgumentParser("FixedGNN")
    # environment setting
    parser.add_argument('--num_agent', type=int, default=3, help='The number of distributed agents/BS.')
    parser.add_argument('--num_antenna', type=int, default=4, help='The number of antennas per BS.')
    parser.add_argument('--num_user', type=int, default=6, help='The number of connected users per BS.')
    parser.add_argument('--channel_mode', type=str, default='mixed_rand', help='The mode of channel simulation. include: mixed_rand, mixed_spec, uni_spec')
    parser.add_argument('--intf_corr', type=int, default=0.5, help='The correlation of interference channels')
    parser.add_argument('--data_corr', type=int, default=0.6, help='The correlation of data channels')
    parser.add_argument('--sigma', type=int, default=1, help='sigma value')
    parser.add_argument('--SNR', type=int, default=20, help='15 [dB]')
    # embedding sizes and layer numbers
    parser.add_argument('--embedding_size', type=int, default=48, help='The embedding size.')
    parser.add_argument('--hidden_state_size', type=int, default=64, help='The hidden state size.')
    parser.add_argument('--num_layer', type=int, default=4, help='The number of MPGNN layers.')
    # leraning
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument("--tau_min", type=float, default=0.1, help="The minimum tau for Gumbel")
    parser.add_argument("--tau_max", type=float, default=10, help="The maximum tau for Gumbel")
    parser.add_argument('--fixed_GNN',type=int,default=1,help='Return True if the GNN arch. is fixed')
    # architecture leraning rate
    # parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    # parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    # log
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
    parser.add_argument('--save_dir', type=str, default='./FixedGNN/', help='Folder to save checkpoints and log.')
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency (default: 200)')
    parser.add_argument('--rand_seed', type=int, default=0, help='manual seed')

    xargs = parser.parse_args([])

    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(xargs.workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(1)

    main_train(xargs)




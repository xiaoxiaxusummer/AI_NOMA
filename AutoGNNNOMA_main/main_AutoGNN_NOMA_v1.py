from train_func import *
from algs.AutoGNN_conv import AutoMPGNN
import torch
from lib.datasets.SearchDataWirelessComm import DatasetNOMA
import math



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main_train(xargs):
    xargs = BuildEnv(xargs)
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(xargs)

    search_model = AutoMPGNN(xargs)
    logger.log('search-model :\n{:}'.format(search_model))
    config_path = 'configs/AutoGNN.config'
    config = load_config(config_path, {'num_agent': xargs.num_agent, 'num_antenna': xargs.num_antenna, 'optim': 'SGD'},
                         logger)
    xargs.batch_size = config.batch_size

    # optimizers of weights and architecture
    w_optimizer, w_scheduler = get_optim_scheduler(search_model.get_weights(), config)
    a_optimizer = torch.optim.Adam(search_model.get_alphas(), lr=xargs.arch_learning_rate, betas=(0.5, 0.999),
                                   weight_decay=xargs.arch_weight_decay)

    logger.log('w-optimizer : {:}'.format(w_optimizer))
    logger.log('a-optimizer : {:}'.format(a_optimizer))
    logger.log('w-scheduler : {:}'.format(w_scheduler))

    # network = torch.nn.DataParallel(search_model,device_ids=[0]).cuda()
    network = search_model.cuda()

    """
    ====================================  automatically resume from previous checkpoint  ====================================
    """
    last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
    if last_info.exists():
        logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))

        ## ================ (1) retrian from last model ================
        last_info = torch.load(last_info)
        start_epoch = last_info['epoch']
        checkpoint = torch.load(last_info['last_checkpoint'])
        ## ================ (2) re-trian from certain epoch ================
        # start_epoch = 6
        # checkpoint = torch.load(str(model_base_path)[:-4]+'_'+str(start_epoch-1)+'.pth')
        ## ================ (3) re-trian from the best model ================
        # checkpoint = torch.load(model_best_path)
        # start_epoch = checkpoint['epoch'] + 1

        valid_comm_costs = checkpoint['valid_comm_costs']
        valid_rates = checkpoint['valid_rates']
        valid_SICs = {}
        search_model.load_state_dict(checkpoint['search_model'])
        w_scheduler.load_state_dict(checkpoint['w_scheduler'])
        w_optimizer.load_state_dict(checkpoint['w_optimizer'])
        a_optimizer.load_state_dict(checkpoint['a_optimizer'])
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        start_epoch, valid_rates, valid_SICs, valid_comm_costs, valid_h_betas, valid_h_archs = 0, {
            'best': -1}, {}, {}, {}, {}

    """
    ================================================  loading datasets  ================================================
    """
    search_data = DatasetNOMA("cluster-free NOMA", xargs, train=True)
    valid_data = DatasetNOMA("cluster-free NOMA", xargs, train=False, saveData=True, small_sample=True)

    search_loader = torch.utils.data.DataLoader(search_data, batch_size=config.batch_size, shuffle=True,
                                                num_workers=1)  # , pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, shuffle=False,
                                               num_workers=1)  # , pin_memory=True)

    logger.log(
        '||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(len(search_loader),
                                                                                     len(valid_loader),
                                                                                     config.batch_size))
    logger.log('||||||| Config={:}'.format(config))

    """
    ====================================================  start training  ==================================================
    """
    start_time, search_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup
    for epoch in range(start_epoch, total_epoch):

        if epoch < 30:
            network.set_tau(
                xargs.tau_max - (xargs.tau_max - xargs.tau_min) * epoch / (30 - 1)
            )
        else:
            network.set_tau(
                xargs.tau_max - (xargs.tau_max - xargs.tau_min) * epoch / (20 * math.ceil(epoch / 20))
            )

        search_data.renew_data()
        search_loader = torch.utils.data.DataLoader(search_data, batch_size=config.batch_size, shuffle=True,
                                                    num_workers=1)  # , pin_memory=True)
        w_scheduler.update(epoch, 0.0)
        need_time = 'Time Left: {:}'.format(convert_secs2time(epoch_time.val * (total_epoch - epoch), True))
        epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
        min_LR = min(w_scheduler.get_lr())
        logger.log('\n[Search the {:}-th epoch] {:}, LR={:}'.format(epoch_str, need_time, min_LR))

        search_w_loss, search_mean_rate, search_comm_cost = search_func(search_loader, network, xargs, w_scheduler,
                                                                        w_optimizer, a_optimizer, epoch_str,
                                                                        xargs.print_freq, logger)
        search_time.update(time.time() - start_time)
        logger.log('[{:}] searching : loss={:.2f}, mean rate={:.2f}%, comm. cost={:.2f}%, time-cost={:.1f} '.format(
            epoch_str, search_w_loss.avg, search_mean_rate, search_comm_cost, search_time.sum))
        valid_start_time = time.time()
        valid_a_loss, valid_rate, valid_SIC, valid_comm_cost = valid_func(valid_loader, network, xargs)
        valid_time = time.time() - valid_start_time
        logger.log(
            '[{:}] evaluate  : loss={:.2f}, rate={:.2f}, SIC={:.2f}, valid_time={:.2f}, comm. cost={:.2f}, lr_arch={:.5f}'.format(
                epoch_str,
                valid_a_loss, valid_rate, valid_SIC, valid_time, valid_comm_cost, xargs.arch_learning_rate))

        valid_rates[epoch] = valid_rate
        valid_SICs[epoch] = valid_SIC
        valid_comm_costs[epoch] = valid_comm_cost

        if valid_rate > valid_rates['best']:
            valid_rates['best'] = valid_rate
            valid_comm_costs['best'] = search_model.get_comm_cost()
            find_best = True
        else:
            find_best = False

        logger.log('<<<--->>> The {:}-th epoch : {:}'.format(epoch_str, valid_comm_costs[epoch]))
        # save checkpoint
        save_path = save_checkpoint({'epoch': epoch + 1,
                                     'args': deepcopy(xargs),
                                     'search_model': search_model.state_dict(),
                                     'w_optimizer': w_optimizer.state_dict(),
                                     'a_optimizer': a_optimizer.state_dict(),
                                     'w_scheduler': w_scheduler.state_dict(),
                                     'valid_comm_costs': valid_comm_costs,
                                     'valid_rates': valid_rates},
                                    model_base_path, logger)

        if epoch % 50 == 0:
            save_checkpoint({'epoch': epoch,
                             'args': deepcopy(xargs),
                             'search_model': search_model.state_dict(),
                             'w_optimizer': w_optimizer.state_dict(),
                             'a_optimizer': a_optimizer.state_dict(),
                             'w_scheduler': w_scheduler.state_dict(),
                             'valid_comm_costs': valid_comm_costs,
                             'valid_rates': valid_rates},
                            str(model_base_path)[:-4] + '_' + str(epoch) + '.pth', logger)

        last_info = save_checkpoint({
            'epoch': epoch + 1,
            'args': deepcopy(xargs),
            'last_checkpoint': save_path,
        }, logger.path('info'), logger)
        if find_best:
            logger.log('<<<--->>> The {:}-th epoch : find the highest validation rate : {:.2f}%.'.format(epoch_str,
                                                                                                         valid_rate))
            copy_checkpoint(model_base_path, model_best_path, logger)
        with torch.no_grad():
            logger.log(
                'arch-parameters :\n{:}'.format(nn.functional.softmax(search_model.arch_parameters, dim=-1).cpu()))
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    logger.log('\n' + '-' * 100)
    # check the performance from the architecture dataset
    logger.log(
        'AutoMPGNN : run {:} epochs, time cost {:.1f} s, last-comm_cost is {:}.'.format(total_epoch, search_time.sum,
                                                                                        valid_comm_costs[
                                                                                            total_epoch - 1]))
    logger.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser("AutoMPGNN")
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
    # log
    parser.add_argument('--save_dir', type=str, default='./result/AutoGNN/',
                        help='Folder to save checkpoints and log.')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency (default: 200)')
    parser.add_argument('--rand_seed', type=int, default=0, help='manual seed')


    xargs = parser.parse_args([])

    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(xargs.workers)


    """
    ===================================== loading models and optimizers  ====================================
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(2)

    main_train(xargs)



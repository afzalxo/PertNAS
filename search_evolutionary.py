import os
import sys
import subprocess
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import copy
import random
from thop import profile
from dataloader.ffcv_cifar10loader import get_ffcv_loaders
from models.model_search import Network
from extras.genotypes import PRIMITIVES
from extras.genotypes import Genotype
from extras import utils
import warnings
from torch.cuda.amp import GradScaler, autocast

warnings.filterwarnings("ignore")

def setup_distributed(rank, local_rank, address, port, cluster, world_size):
    os.environ['MASTER_ADDR'] = address
    os.environ['MASTER_PORT'] = port
    print('Setting up dist training rank %d' % rank)
    dirname = os.path.dirname(__file__)
    if cluster == 'local':
        init_method = 'file://' + dirname + '/dist_communication_file'
    else:
        init_method = 'env://'
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--workers', type=int, default=12, help='number of workers to load dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.2176, help='init learning rate') #0.025
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate') #0.001
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=5, help='total number of layers')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--save', type=str, help='experiment path')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--finetune_steps', type=int, default=50, help='number of steps to finetune before measuring perturb acc')
    parser.add_argument('--train_path', type=str, default='/home/aahmadaa/datasets/cifar_ffcv', help='Location of FFCV CIFAR dataset')
    parser.add_argument('--note', type=str, default='try', help='note for this run')
    parser.add_argument('--cifar100', action='store_true', default=False, help='search with cifar100 dataset')
    parser.add_argument('--op_cap', type=int, default='1', help='Number of operations to cap per edge')
    parser.add_argument('--fast', action='store_true', default=False, help='eval/train on one batch, for debugging')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Use W&B?')
    parser.add_argument('--npop', type=int, default='4', help='Number of models in population')
    parser.add_argument('--nodes', type=int, default='4', help='Number of intermediate nodes per cell')
    parser.add_argument('--cluster', type=str, default='local', help='Where to execute?')
    parser.add_argument('--distributed', action='store_true', default=True, help='Distributed search?')

    args = parser.parse_args()
    if args.distributed and args.cluster == 'tacc':
        global_rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        iplist = os.environ['SLURM_JOB_NODELIST']
        ip = subprocess.getoutput(f'scontrol show hostname {iplist} | head -n1')
    elif args.distributed and args.cluster == 'local':
        local_rank = int(os.environ['LOCAL_RANK'])
        global_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        ip = '127.0.0.1'
    else:
        local_rank = 0
        world_size = 1
        global_rank = 0
        ip = 'localhost'

    args.world_size = world_size
    args.global_rank = global_rank
    args.local_rank = local_rank
    args.ip = ip
    args.val_batch_size = int(args.batch_size * 1.0)
    args.save = '{}search-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
    if args.distributed: 
        setup_distributed(args.global_rank, args.local_rank, ip, str(33182), args.cluster, args.world_size)
        dist.barrier()
    if global_rank == 0:
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    if args.use_wandb:
        wandb_identity = '166a45fa2ad2b2db9ec555119b273a3a9bdacc41'
        import wandb
        os.environ["WANDB_API_KEY"] = wandb_identity
        os.environ["WANDB_ENTITY"]='europa1610'
        os.environ["WANDB_PROJECT"]='PertNAS-FFCV'
        wandb_dir = args.save
        wandb_con = wandb.init(project='PertNAS-FFCV', entity='europa1610', dir=wandb_dir, name=args.note + '_' + str(local_rank), group='clustered-search')

    #Different color schemes for log output from different processes
    color_purple = '\033[1;35;48m'
    color_green = '\033[1;32;48m'
    color_cyan = '\033[1;36;48m'
    color_yellow = '\033[1;33;48m'
    color_reset = '\033[1;37;0m'
    colors = [color_purple, color_green, color_cyan, color_yellow, color_reset]

    log_format = '%(asctime)s %(message)s'
    log_format = colors[global_rank] + log_format
    log_format = log_format + color_reset

    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    if args.distributed:
        dist.barrier()

    fh = logging.FileHandler(os.path.join(args.save, f'log_p{local_rank}.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    CIFAR_CLASSES = 100 if args.cifar100 else 10
        
    start_time = time.time()

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)

    if args.use_wandb:
        wandb.config.update(args) 
        wandb.save('./*.py')
        args.wandb_con = wandb_con

    ### FFCV loader here
    train_queue, valid_queue = get_ffcv_loaders(args.train_path, args.batch_size, args.val_batch_size, local_rank)
    criterion = nn.CrossEntropyLoss().to(f'cuda:{local_rank}')
    for n_iter in range(args.npop // world_size):
        start_ep = 0
        switches_normal = _init_switches(args.op_cap)
        logging.info('Iteration %d, Initial switches_normal: %s', n_iter, switches_normal)

        model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, steps=4, multiplier=4, switches_normal=switches_normal)
        input = torch.randn(1, 3, 224, 224)
        macs, params = profile(model, inputs=(input, ))
        print('MFLOPS: {}, MPARAMS: {}'.format(macs/1000000, params/1000000))
        #model = nn.DataParallel(model)
        model = model.to(memory_format=torch.channels_last)
        model = model.to(f'cuda:{local_rank}')

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        network_params = []
        for k,v in model.named_parameters():
            network_params.append(v)       
        optimizer = torch.optim.SGD(network_params, args.learning_rate,momentum=args.momentum, weight_decay=args.weight_decay)
        _epochs_pretrain = 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(_epochs_pretrain), eta_min=args.learning_rate_min)
        
        _old_weight_storage = dict()
        _total_edges = 14
        _total_nodes = 4
        _edges = [i for i in range(_total_edges)]
        _nodes = [i+1 for i in range(_total_nodes-1)]
        done_expand_perturb = False
        _done_top = False
        _done_op_selection = False
        _validation_begin_epoch = _epochs_pretrain-2
        _perturb_every = 5
        _perturb_every_op_top = 5
        _greedy_select_train_epochs = 5
        _edges_discretized = 0
        valuation_matrix = np.array([[0. for j in range(len(PRIMITIVES))] for i in range(_total_edges)])

        switch_store = None
        while not done_expand_perturb:
            #_edge = _edges[0]
            _edge = _edges[0]
            _edges.remove(_edge)
            _old_state_dict = model.state_dict()
            switches_normal, switch_store = update_switches(switches_normal, edge=_edge, switch_store=switch_store)
            logging.info('Edge %d, Normal Switches %s, Switch Store %s:' % (_edge, str(switches_normal), str(switch_store)))
            if len(_edges) == 0:
                _edges = [i for i in range(_total_edges)]
                done_expand_perturb = True
            del model
            #Construct new model out of old params and new switches
            model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, steps=4, multiplier=4, switches_normal=switches_normal) 
            model = model.to(memory_format=torch.channels_last)
            model = model.to(f'cuda:{local_rank}')
            #_new_state_dict = model.state_dict()
            #_new_state_dict, _old_weight_storage = utils._restore_weights(_old_state_dict, _new_state_dict, _old_weight_storage)
            #model.load_state_dict(_new_state_dict)
            network_params = []
            for k,v in model.named_parameters():
                network_params.append(v)
            optimizer = torch.optim.SGD(network_params, args.learning_rate/2.5, momentum=args.momentum, weight_decay = args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(_perturb_every), eta_min=args.learning_rate_min)
            train_x_epochs(_perturb_every, 9999, scheduler, train_queue, valid_queue, model, criterion, optimizer, args)
            logging.info('Checking Edge %d, Edges %s', _edge, str(_edges))
            edge_imp_vec = perturb_val(model, criterion, optimizer, network_params, scheduler.get_last_lr()[0], finetune_steps=args.finetune_steps+20, args=args, rank=local_rank, _edge=_edge)
            logging.info('Edge Values %s', str(np.array(edge_imp_vec)))
            # Update valuation matrix for this model in the population
            valuation_matrix = update_valuation_matrix(edge_imp_vec, valuation_matrix, switches_normal, edge=_edge)
            logging.info('Rank %d, Valuation Matrix:\n %s', local_rank,str(valuation_matrix))
        
        share_exp_ret = torch.tensor(valuation_matrix).to(local_rank)
        if n_iter == 0:
            running_avg_tensor_list = []
        if args.distributed:
            dist.barrier()
            dist.all_reduce(share_exp_ret, op=dist.ReduceOp.SUM)
            share_exp_ret = share_exp_ret / world_size
            running_avg_tensor_list.append(share_exp_ret)
            tensor_sum = torch.zeros_like(share_exp_ret)
            for p, ten in enumerate(running_avg_tensor_list):
                tensor_sum += ten
            cur_avg_val = tensor_sum / len(running_avg_tensor_list)
            logging.info('Iter: %d, Current Average Value: %s', n_iter, str(cur_avg_val))
            dist.barrier()
    valuation_matrix = cur_avg_val.detach().cpu().numpy()
    print('===='*10)
    if args.distributed:
        #print('PID: {}, Finished in {}'.format(pid, time.time()-start_time))
        print('Finished process..')
        dist.barrier()
    if global_rank == 0:
        #Do partial supernet selection
        logging.info('Selecting top-2 ops per edge...')
        switches_normal = coarse_grained_op_select(valuation_matrix)
        logging.info('Coarse grained selection finished:\n switches_n: %s', str(switches_normal))

        # Count number of switches > 2 to evaluate whether enough conv ops survived
        # This part is for testing and can be removed
        convs_norm = 0
        for _ed in range(len(switches_normal)):
            for _op in range(len(switches_normal[_ed])):
                if switches_normal[_ed][_op] > 2:
                    convs_norm += 1
        logging.info('Number of convs in normal cell: %d', convs_norm)

        _edges = [i for i in range(_total_edges)]
        del model
        model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, steps=4, multiplier=4, switches_normal=switches_normal) 
        model = model.to(memory_format=torch.channels_last)
        model = model.to(f'cuda:{local_rank}')
        network_params = []
        for k,v in model.named_parameters():
            network_params.append(v)
        optimizer = torch.optim.SGD(network_params, args.learning_rate/3., momentum=args.momentum, weight_decay = args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(_greedy_select_train_epochs), eta_min=args.learning_rate_min)
        train_x_epochs(_greedy_select_train_epochs, 0, scheduler, train_queue, valid_queue, model, criterion, optimizer, args)

        #Do operation selection
        _edges_discretized = 0
        _edges = [13, 12, 8, 11, 7, 4, 9, 10, 5, 6, 2, 3, 0, 1]
        while not _done_op_selection:
            _edge = _edges[0]
            _edges.remove(_edge)
            import_mat = perturb_val(model, criterion, optimizer, network_params, scheduler.get_last_lr()[0], finetune_steps=args.finetune_steps+80, args=args, rank=local_rank, _edge=_edge)
            logging.info('Checking Edge %d, Edges %s', _edge, str(_edges))
            logging.info('Edge %d, Edge Importance %s', _edge, str(np.array(import_mat)))
            _old_state_dict = model.state_dict()
            logging.info('Fine-Grained Operation Selection, Edge %d, Discretized %d edges so far...' % (_edge, _edges_discretized))
            switches_normal = fine_grained_op_select(import_mat, _edge, switches_normal)
            logging.info('Discretized Edge %d, Normal Switches %s' % (_edge, str(switches_normal)))

            _edges_discretized += 1
            if len(_edges) == 0:
                _edges = [i for i in range(_total_edges)]
                _done_op_selection = True
                for _e in range(len(switches_normal)):
                    switches_normal[_e] = switches_normal[_e][:-1]
                logging.info('Fined-grained op selection finished, Final Ops: %s', str(switches_normal))
            del model
            model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, steps=4, multiplier=4, switches_normal=switches_normal) 
            model = model.to(memory_format=torch.channels_last)
            model = model.to(f'cuda:{local_rank}')
            #_new_state_dict = model.state_dict()
            #_new_state_dict, _old_weight_storage = utils._restore_weights(_old_state_dict, _new_state_dict, _old_weight_storage)
            #model.load_state_dict(_new_state_dict)
            network_params = []
            for k,v in model.named_parameters():
                network_params.append(v)
            optimizer = torch.optim.SGD(network_params, args.learning_rate/2.5, momentum=args.momentum, weight_decay = args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(_perturb_every_op_top), eta_min=args.learning_rate_min)
            train_x_epochs(_perturb_every_op_top, 0, scheduler, train_queue, valid_queue, model, criterion, optimizer, args)
    if global_rank == 0:
        #Topology selection starts here
        while not _done_top:
            optimizer = torch.optim.SGD(network_params, args.learning_rate/2.5, momentum=args.momentum, weight_decay = args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(_perturb_every_op_top), eta_min=args.learning_rate_min)
            import_mat = perturb_val(model, criterion, optimizer, network_params, scheduler.get_last_lr(), finetune_steps=args.finetune_steps+130, include_none=False, args=args, rank=local_rank, _edge=None)
            _old_state_dict = model.state_dict()
            _node = random.choice(_nodes)
            _nodes.remove(_node)
            logging.info('Applying topology selection on node %d...', _node)
            logging.info('Importance Matrix for Topology Selection on Node %d is: %s', _node, str(np.array(import_mat)))
            switches_normal, final_topology_n = discretize_node(import_mat, switches_normal, _node)
            logging.info('Current topology switches_n: %s, topology normal: %s', switches_normal, final_topology_n)
            if len(_nodes) == 0:
                _done_top = True
                logging.info('Topology selection finished...')
                logging.info('genotype=%s', topology_to_genotype(switches_normal))
            del model
            #Construct new model out of old params and new switches
            model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, steps=4, multiplier=4, switches_normal=switches_normal) 
            model = model.to(memory_format=torch.channels_last)
            model = model.to(f'cuda:{local_rank}')
            #if not _done_top:
            #    _new_state_dict = model.state_dict()
            #    _new_state_dict, _old_weight_storage = utils._restore_weights(_old_state_dict, _new_state_dict, _old_weight_storage)
            #    model.load_state_dict(_new_state_dict)
            network_params = []
            for k,v in model.named_parameters():
                network_params.append(v)
            if not _done_top:
                optimizer = torch.optim.SGD(network_params, args.learning_rate/2.5, momentum=args.momentum, weight_decay = args.weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(_perturb_every_op_top), eta_min=args.learning_rate_min)
                train_x_epochs(_perturb_every_op_top, 0, scheduler, train_queue, valid_queue, model, criterion, optimizer, args)

def train_x_epochs(epochs, validation_begin_epoch, scheduler, train_queue, valid_queue, model, criterion, optimizer, args):
    scaler = GradScaler()
    for epoch in range(epochs):
        epoch_start = time.time()
        # training
        lr = scheduler.get_last_lr()[0]
        logging.info('Epoch: %d lr: %f', epoch, lr)
        if args.use_wandb:
            args.wandb_con.log({'Learning Rate': lr}, commit=False)
        logging.info('GPU Memory Acclocated: %s, Max Alloc %s, Mem Reserved %s, Max Mem Reserved %s', torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved(), torch.cuda.max_memory_reserved())
        train_acc, train_obj = train(train_queue, valid_queue, model, criterion, optimizer, scaler=scaler, args=args)
        scheduler.step()
        logging.info('Train_acc %f', train_acc)
        epoch_duration = time.time() - epoch_start
        logging.info('Epoch train time: %ds', epoch_duration)
        if args.use_wandb:
            args.wandb_con.log({'Train Accuracy': train_acc, 'Train Loss': train_obj}, commit=False)
        # validation
        if epoch >= validation_begin_epoch:
            valid_acc, valid_obj = infer(valid_queue, model, criterion, args=args)
            logging.info('Epoch %d, Valid_acc %f, Valid_loss %f', epoch, valid_acc, valid_obj)
            if args.use_wandb:
                args.wandb_con.log({'Validation Accuracy': valid_acc, 'Validation Loss': valid_obj}, commit=False)
        if args.use_wandb:
            args.wandb_con.log({'epoch': epoch})


def perturb_val(model, criterion, optimizer, network_params, lr, finetune_steps=None, include_none=True, args=None, rank=None, _edge=None):
    logging.info('Rank %d, Applying perturb-val to edge %s with finetune_steps %d...', rank, str(_edge), finetune_steps)
    nodes = 4
    k = sum(1 for i in range(nodes) for n in range(2+i))
    acp=time.time()
    _sd_backup = model.state_dict()
    _optim_sd_backup = optimizer.state_dict()
    t_queue, v_queue = get_ffcv_loaders(args.train_path, args.batch_size, args.val_batch_size, rank)
    acc_before, _ = infer(v_queue, model, criterion, log_off=True, args=args)
    scaler = GradScaler()
    if _edge == None:
        perturb_acc_diff = [[0 for x in range(len(model.arch_parameters()[0][e])-include_none)] for e in range(k)]
        for edge in range(k):
            for ops in range(len(model.arch_parameters()[0][edge])-include_none):
                loader_stime = time.time()
                #t_queue, v_queue = get_ffcv_loaders(args.train_path,args.batch_size, args.val_batch_size, rank)
                model._perturb_binary(edge, ops, remove=True)

                #Fine-tune perturbed arch
                if not args.fast and finetune_steps != 0:
                    model.load_state_dict(_sd_backup)
                    optimizer.load_state_dict(_optim_sd_backup)
                    train(t_queue, v_queue, model, criterion, optimizer, finetune_steps=finetune_steps, partial=True, scaler=scaler, args=args)
                acc_after, _ = infer(v_queue, model, criterion, log_off=True, args=args)

                model._perturb_binary(edge, ops, remove=False)
                perturb_acc_diff[edge][ops] = acc_after - acc_before
                print('Edge %d, Op %d, Acc Before %f, Acc After %f, Perturb Acc Diff %f, Time %s' % (edge, ops, acc_before, acc_after, perturb_acc_diff[edge][ops], str(time.time() - loader_stime)))
    else:
        perturb_acc_diff = [0. for x in range(len(model.arch_parameters()[0][_edge])-include_none)]
        for ops in range(len(model.arch_parameters()[0][_edge])-include_none):
            loader_stime = time.time()
            #t_queue, v_queue = get_ffcv_loaders(args.train_path, args.batch_size, args.val_batch_size, rank)
            model._perturb_binary(_edge, ops, remove=True)

            #Fine-tune perturbed arch
            if not args.fast and finetune_steps != 0:
                model.load_state_dict(_sd_backup)
                optimizer.load_state_dict(_optim_sd_backup) 
                logging.info('Finetuning for %s steps...', str(finetune_steps))
                train(t_queue, v_queue, model, criterion, optimizer, finetune_steps=finetune_steps, partial=True, scaler=scaler, args=args)
            acc_after, _ = infer(v_queue, model, criterion, log_off=True, args=args)

            model._perturb_binary(_edge, ops, remove=False)
            perturb_acc_diff[ops] = acc_after - acc_before
            print('Edge %d, Op %d, Acc Before %f, Acc After %f, Perturb Acc Diff %f, Time %s' % (_edge, ops, acc_before, acc_after, perturb_acc_diff[ops], str(time.time() - loader_stime)))

    model.load_state_dict(_sd_backup)
    optimizer.load_state_dict(_optim_sd_backup)
    logging.info('Perturb Val Time Taken: %ds', time.time()-acp)
    return perturb_acc_diff

def topology_to_genotype(topology_n):

    nodes = 4
    def _parse(topology):
        gene = []
        n = 2
        start = 0
        for _node in range(nodes):
            end = start+n
            ed_top = topology[start:end].copy()
            for i in range(len(ed_top)):
                if ed_top[i][0] != 100:
                    gene.append((PRIMITIVES[ed_top[i][0]], i))
            start = end
            n += 1
        return gene
    gene_normal = _parse(topology_n)
    concat = range(2, 2+nodes)
    genotype = Genotype(normal=gene_normal, normal_concat=concat, reduce=gene_normal, reduce_concat=concat)
    return genotype

def coarse_grained_op_select(valuation_matrix):
    switches_n = [sorted(np.argsort(valuation_matrix[e])[:2].tolist()+[100]) for e in range(len(valuation_matrix))]
    return switches_n

def update_valuation_matrix(imp, valuation_matrix, switches_n, edge):
    valuation_matrix = np.array(valuation_matrix)
    valuation_matrix[edge][np.array(switches_n[edge][:-1])] = np.array(imp)
    return valuation_matrix

def fine_grained_op_select(imp_f, edge, switches_n):
    _switches_n = copy.deepcopy(switches_n)
    logging.info('Initiating op selection on Edge %d with Switches_n %s', edge, str(switches_n))
    sample_args_n = np.argsort(imp_f)[:1]
    _switches_n[edge] = sorted(np.array(_switches_n[edge])[sample_args_n].tolist()+[100], reverse=False)
    logging.info('Selected op(s) normal for edge %d are %s', edge, str(_switches_n[edge]))
    return _switches_n

def update_switches(switches_n, edge, switch_store=None): 
    _switches_n = copy.deepcopy(switches_n)
    if switch_store is not None and edge != 0:
        _switches_n[edge-1] = switch_store
    switch_store = _switches_n[edge]
    _switches_n[edge] = [i for i in range(len(PRIMITIVES))]
    _switches_n[edge].append(100)
    return _switches_n, switch_store

def discretize_node(imp, switches_n, _node):
    assert(len(switches_n[0]) == 1)
    _imp_n = imp
    switches_n = copy.deepcopy(switches_n)
    num_edg = 14
    n = 2
    start = 0
    step = 4
    for i in range(_node):
        end = start+n
        start = end
        n += 1
    end = start+n

    arr_n = _imp_n[start:end]
    sec_ma_n, ma_n = np.argsort(np.min(arr_n, axis=1))[:2]
    for j in range(start, end):
        if j not in [start+ma_n, start+sec_ma_n]:
            switches_n[j] = [100]
    _nswitches = []
    for j in range(len(switches_n)):
        _nswitches.append(switches_n[j])

    switches_n = _nswitches
    final_arch_n = []
    final_arch_n.append([PRIMITIVES[switches_n[i][0]] if switches_n[i][0] != 100 else 'none' for i in range(len(switches_n))])

    return switches_n, final_arch_n

def train(train_queue, valid_queue, model, criterion, optimizer, finetune_steps=None, partial=False, scaler=None, args=None):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    
    model.train()
    for step, (input, target) in enumerate(train_queue):
        if args.fast and step > 0:
            break
        n = input.size(0)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(input)
            loss = criterion(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        prec1, prec5 = utils.accuracy(logits.detach(), target.detach(), topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0 and not partial:
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, objs.avg, top1.avg, top5.avg)
        if partial and step >= finetune_steps:
            break
        del loss, logits

    return top1.avg, objs.avg

def infer(valid_queue, model, criterion, log_off=False, args=None):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            if args.fast and step > 0:
                break
            #input = input.cuda()
            #target = target.cuda(non_blocking=True)
            with autocast():
                logits = model(input)
                loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits.detach(), target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if not log_off:
                if step % args.report_freq == 0:
                    logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            del loss, logits

    return top1.avg, objs.avg

def _init_switches(_op_cap):
    switches = []
    nodes = 4
    k = sum(1 for i in range(nodes) for n in range(2+i))
    for i in range(k):
        switches.append(sorted(random.sample(range(len(PRIMITIVES)), _op_cap)+[100], reverse=False))
        #switches.append([3])
    return switches

if __name__ == '__main__':
    start_time = time.time()
    main() 
    duration = time.time() - start_time
    logging.info('Total searching time: %ds', duration)

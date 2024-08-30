import copy
import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb

from arguments import args, logger
import numpy as np
from torch_geometric.loader import DataLoader
from layouts import generate_layouts
from off_loading_models import TaskLoad, PCNet
from tqdm import tqdm
import random
import os
import time
from func import loss_fn_kd
# from config import get_config
import math


def compute_loss_nn(task_allocation, power_allocation, comp_allocation, task_size, compute_resource, path_losses,
                    user_index, server_index, tg = None, pg = None, cg = None, mode = 'avg'):
    # task_size : vector N
    # task_allocation: mat pop_num x 3*M*N
    # index: vector 3*M*N

    epsilon = 1e-9
    extre = 1e-20
    extre = args.extre
    # extre = 1e-12
    server_index_first = server_index.reshape((batch_size, -1))[0]
    user_index_first = user_index.reshape((batch_size, -1))[0]
    # user_index = edge_index[0]      # s2u中源节点的索引
    # server_index = edge_index[1]    # s2u中目标节点的索引

    # power_allocation = torch.clamp(power_allocation, 1e-5, 1)
    pw_ini = power_allocation * path_losses  # mat pop_num x M*N
    # pw = torch.clamp(pw, 1e-5, 1)

    # 将信道状态过小的设置为0
    mask_pw = torch.where(pw_ini < args.pw_threshold)

    pw = pw_ini.clone()

    pw[mask_pw] = 0

    comp_allocation_clone = comp_allocation.clone()
    comp_allocation_clone[mask_pw] = 0
    comp_allocation_normed = torch.zeros(
        (comp_allocation_clone.shape[0], comp_allocation_clone.shape[1], server_index_first[-1] + 1),
        device = args.device)
    comp_allocation_normed.scatter_(2, server_index_first.repeat((batch_size, 1)).unsqueeze(2),
                                    comp_allocation_clone.unsqueeze(2))
    comp_allocation_normed = comp_allocation_normed.sum(1)[:, server_index_first]
    comp_allocation_normed = torch.div(comp_allocation_clone, comp_allocation_normed + extre)

    task_allocation_clone = task_allocation.clone()
    task_allocation_clone[mask_pw] = 0
    task_allocation_normed = torch.zeros(
        (task_allocation_clone.shape[0], task_allocation_clone.shape[1], user_index_first[-1] + 1),
        device = args.device)
    task_allocation_normed.scatter_(2, user_index_first.repeat((batch_size, 1)).unsqueeze(2),
                                    task_allocation_clone.unsqueeze(2))
    task_allocation_normed = task_allocation_normed.sum(1)[:, user_index_first]
    task_allocation_normed = torch.div(task_allocation_clone, task_allocation_normed + extre)

    # 计算速率
    pw_list = torch.zeros((pw.shape[0], pw.shape[1], server_index_first[-1] + 1),
                          device = args.device)  # mat pop_num x MN x N
    pw_list.scatter_(2, server_index_first.repeat((batch_size, 1)).unsqueeze(2), pw.unsqueeze(2))
    pws_list = pw_list.sum(1)[:, server_index_first]  # mat pop_num x MN

    interference = pws_list - pw
    rate = torch.log2(1 + torch.div(pw, interference + epsilon))
    # rate = args.band_width * torch.log2(1+torch.div(pw, interference+epsilon))

    task_size = task_size[:, user_index_first]  # M*N
    # task_size = task_size[user_index]*args.tasksize_cof       # 重复采样映射到边中
    tasks = task_size * task_allocation_normed  # mat pop_num x M*N

    compute_resource = compute_resource[:, server_index_first]
    # compute_resource = compute_resource[server_index]*args.comp_cof       #

    comp = compute_resource * comp_allocation_normed

    # offloading_time = torch.div(tasks, rate+extre) * (args.tasksize_cof/args.band_width)
    offloading_time = torch.div(tasks, rate + extre)

    # compute_time = torch.div(tasks, comp+extre) * (args.tasksize_cof*args.cons_factor/args.comp_cof)
    compute_time = torch.div(tasks, comp + extre)

    time_loss = offloading_time + compute_time  # pop_num x MN
    assert torch.isnan(time_loss).sum() == 0

    time_loss_list = torch.zeros((time_loss.shape[0], time_loss.shape[1], user_index_first[-1] + 1),
                                 device = args.device)
    time_loss_list.scatter_(2, user_index_first.repeat((batch_size, 1)).unsqueeze(2), time_loss.unsqueeze(2))
    time_loss_list = time_loss_list.sum(1)  # pop_num x MN

    if mode == 'avg' or mode == 'prox':
        assert tg is None
        return time_loss_list.mean()
    elif mode == 'kd':
        assert tg is not None
        beta = args.beta
        T = args.tem
        total_KD_loss = 0
        student_res = [task_allocation, power_allocation, comp_allocation]
        teacher_res = [tg, pg, cg]
        for st, tt in zip(student_res, teacher_res):
            median_value = 1 / len(teacher_res) * nn.KLDivLoss()(F.log_softmax(st / T, dim = 1),
                                                                 F.softmax(tt / T, dim = 1)) * (beta * T * T)
            total_KD_loss += median_value

        return time_loss_list.mean() * (1 - beta) + total_KD_loss

minimum_threshold = 1e-27
# minimum_threshold = 1e-3
def judge(temp):
    res = torch.where(temp < minimum_threshold, temp - torch.inf, temp)
    return res


def client_train(i, model, global_model, graph, time_step, args, step = 1, pretrain = False):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = nn_lr)
    global_model.eval()
    for st in range(step):
        u2s_index = graph['user', 'u2s', 'server'].edge_index
        user_index = u2s_index[0]
        server_index = u2s_index[1]
        u2s_path_loss = graph['user', 'u2s', 'server'].path_loss.squeeze().reshape((batch_size, -1))
        u2s_path_loss_feat = graph['user', 'u2s', 'server'].edge_attr.squeeze().reshape((batch_size, -1))
        user_tasksize = graph['user'].x[:, 0].reshape((batch_size, -1))
        server_comp_resource = graph['server'].x[:, 0].reshape((batch_size, -1))

        output_len = args.server_list[i] * args.user_list[i]
        task_sche, power_sche, comp_sche = \
            model(u2s_path_loss_feat, user_tasksize, server_comp_resource, u2s_index, output_len)

        if args.flMode == 'kd':
            tg, pg, cg = global_model(u2s_path_loss_feat, user_tasksize, server_comp_resource, u2s_index, output_len)
        else:
            tg, pg, cg = None, None, None

        loss = compute_loss_nn(task_sche, power_sche, comp_sche, user_tasksize, server_comp_resource, u2s_path_loss,
                               user_index, server_index, tg, pg, cg, args.flMode)

        if args.flMode == 'prox':
            proximal_term = 0.0
            for w_client, w_global in zip(model.parameters(), global_model.parameters()):
                proximal_term += ((w_client - w_global) ** 2).sum()
            loss += args.prox_coef * proximal_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



# 联邦学习
def local_train(gnn_train_loader, global_model, time_step, user_models, args, pretrain = False):
    for i, graph in enumerate(gnn_train_loader):
        client_train(i, user_models[i], global_model, graph, time_step, step = args.inner_step, pretrain = pretrain, args = args)


# 聚合模型参数
def aggregate_parameters(user_models, global_model, mode = 'avg'):
    # global_model = copy.deepcopy(user_models[0].eval())

    for param in global_model.parameters():
        param.data = torch.zeros_like(param.data)

    if mode == 'n_encoder_mlp':
        for key in global_model.state_dict().keys():
            if 'convs' or 'encoder' in key:
                for user in user_models:
                    global_model.state_dict()[key] += user.state_dict()[key] / len(user_models)
        return global_model

    for user in user_models:
        for gp, lp in zip(global_model.parameters(), user.parameters()):
            gp.data += lp.data / len(user_models)
    return global_model


# def NN_train(train_loader, test_loader, global_model, user_models):
def NN_train(global_model, user_models, train_loader):
    policy_losses = []
    test_policy_losses = []

    round = 1
    change_dir = True
    t = time.localtime()
    path = './TO_models/NN/' + multi_scales + '/{}-{}-{}/r{}beta{}tem{}epoch{}lr{}lrf{}'.format(
        t.tm_year, t.tm_mon, t.tm_mday, round, args.beta, args.tem, args.epoch, args.pcnet_lr, args.lrf)

    thre = args.thre
    global best_loss
    best_loss = 10086
    save_count = 0

    # 增加 scheduler
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epoch)) / 2) * (1 - args.lrf) + args.lrf
    global nn_lr
    nn_lr = args.pcnet_lr

    logger.info(
        'beta = {}, lr = {}, epoch = {}, server_range = [{}, {}]'.format(args.beta, args.pcnet_lr, args.epoch, args.min,
                                                                         args.max))
    if args.use_wandb:
        # wandb sweep
        config = dict(
            tem = args.tem,
            beta = args.beta,
            epoch = args.epoch,
            lr = args.pcnet_lr,
            lrf = args.lrf
        )
        wandb.init(config = config,
                   project = args.wandb_project,
                   entity = 'xdu_ai',
                   name = args.mlp_wandb_name
                   )

    # schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.9)
    loss_dict = {f'graph_{i}': [] for i in range(len(nn_eval_loader))}
    for time_step in tqdm(range(Epochs)):
        loss_sum = 0
        length = 0

        if args.use_lrScheduler:
            nn_lr = lf(time_step) * args.pcnet_lr

        if args.mode == 'n_encoder_mlp':
            for key in global_model.state_dict().keys():
                if 'convs' or 'encoder' in key:
                    for i in range(len(user_models)):
                        user_models[i].state_dict()[key] = global_model.state_dict()[key]
            local_train(train_loader, global_model, time_step, user_models, args = args, pretrain = False)
        else:
            for i in range(len(user_models)):
                user_models[i].load_state_dict(global_model.state_dict())
            local_train(train_loader, global_model, time_step, user_models, args = args, pretrain = False)

        global_model = aggregate_parameters(user_models, global_model, args.mode)
        global_model.eval()

        temp_loss_dict = {}

        eval_loader_dict = {
            'globalModelOnEachGraph': train_loader,
            'trainOnlyOneGraph': nn_eval_loader
        }
        if eval_loader_dict.get(args.wandb_project):
            eval_loader = eval_loader_dict[args.wandb_project]
        else:
            eval_loader = train_loader

        # for i, graph in enumerate(train_loader):  # graph为一个batch
        for i, graph in enumerate(eval_loader):  # graph为一个batch
            u2s_index = graph['user', 'u2s', 'server'].edge_index
            user_index = u2s_index[0]
            server_index = u2s_index[1]
            u2s_path_loss = graph['user', 'u2s', 'server'].path_loss.squeeze().reshape((batch_size, -1))
            u2s_path_loss_feat = graph['user', 'u2s', 'server'].edge_attr.squeeze().reshape((batch_size, -1))
            user_tasksize = graph['user'].x[:, 0].reshape((batch_size, -1))
            server_comp_resource = graph['server'].x[:, 0].reshape((batch_size, -1))
            output_len = args.eval_server_list[i] * args.eval_user_list[i]

            if args.mode == 'n_encoder_mlp':
                # 有问题！！！！
                task_allocation, power_allocation, comp_allocation = user_models[i](graph.x_dict, graph.edge_index_dict,
                                                                                    graph.edge_attr_dict)
            else:
                task_sche, power_sche, comp_sche = global_model(
                    u2s_path_loss_feat, user_tasksize, server_comp_resource, u2s_index, output_len)

            loss_batch = compute_loss_nn(task_sche, power_sche, comp_sche, user_tasksize, server_comp_resource,
                                   u2s_path_loss, user_index, server_index)
            # loss_batch = compute_loss_nn(task_allocation, power_allocation, comp_allocation, graph['server'].x[:, 0],
            #                           graph['user', 'u2s', 'server'].path_loss, graph['user'].x[:, 0], user_index,
            #                           server_index, mode = args.flMode)
            loss_dict[f'graph_{i}'].append(loss_batch.item())
            temp_loss_dict[f'graph_{i}'] = loss_batch.item()
            loss_sum += loss_batch.item()
            length += 1
        policy_loss = loss_sum / length

        # if (time_step + 1) % save_fre == 0:
        if policy_loss < best_loss:
            best_loss = policy_loss
            save_count = 0
            if not os.path.exists(path):
                os.makedirs(path)
                change_dir = False
            elif change_dir:
                while os.path.exists(path):
                    round += 1
                    path = './TO_models/NN/' + multi_scales + '/{}-{}-{}/r{}beta{}tem{}epoch{}lr{}lrf{}'.format(
                        t.tm_year, t.tm_mon, t.tm_mday, round, args.beta, args.tem, args.epoch, args.pcnet_lr, args.lrf)
                os.makedirs(path)
                change_dir = False
            # torch.save(global_model, os.path.join(path, 'HGNN_{}_{}_{}.pt'.format(train_num_layouts, time_step + 1, policy_loss)))
            if policy_loss < thre:
                torch.save(global_model.state_dict(),
                           os.path.join(path, 'NN_{}_{}_{}.pt'.format(train_num_layouts, policy_loss, time_step + 1)))
        else:
            save_count += 1
        # if save_count >= 250 and policy_loss < thre:
        if save_count >= args.max_save_interval:
            save_count = 0
            ano_path = './TO_models/NN/{}/{}-{}-{}/r{}beta{}tem{}epoch{}lr{}lrf{}/long_time_no_save'.format(
                multi_scales, t.tm_year, t.tm_mon, t.tm_mday, round, args.beta, args.tem, args.epoch, args.pcnet_lr,
                args.lrf)
            if not os.path.exists(ano_path):
                os.makedirs(ano_path)
            # torch.save(global_model, os.path.join(ano_path, 'HGNN_{}_{}_{}.pt'.format(train_num_layouts, policy_loss, time_step + 1)))
            torch.save(global_model.state_dict(),
                       os.path.join(ano_path, 'NN_{}_{}_{}.pt'.format(train_num_layouts, policy_loss, time_step + 1)))

        # policy_losses.append(policy_loss)
        logger.info('step=={}, policy_loss=={}, lr=={}'.format(time_step, policy_loss, nn_lr))
        if args.use_wandb:
            temp_loss_dict['mean_loss'] = float(policy_loss)
            wandb.log(temp_loss_dict)
            # if args.wandb_project == 'globalModelOnEachGraph':
            #     temp_loss_dict['mean_loss'] = float(policy_loss)
            #     wandb.log(temp_loss_dict)
            # elif 'FL_LOGNN' in args.wandb_project:
            #     wandb.log(dict(policy_loss = float(policy_loss)))
            # elif 'trainOnlyOneGraph' in args.wandb_project:
            #
            # else:
            #     print('unknown wandb project')
            #     exit(1)

    if args.use_wandb:
        wandb.finish()
    return global_model, np.array(policy_losses), np.array(test_policy_losses)


def NN_eval(step, loader, model):
    sum_loss = 0
    length = 0
    model.eval()
    for idx, batch in enumerate(loader):
        u2s_index = batch['user', 'u2s', 'server'].edge_index
        user_index = u2s_index[0]
        server_index = u2s_index[1]
        u2s_path_loss = batch['user', 'u2s', 'server'].path_loss.squeeze().reshape((batch_size, -1))
        u2s_path_loss_feat = batch['user', 'u2s', 'server'].edge_attr.squeeze().reshape((batch_size, -1))
        user_tasksize = batch['user'].x[:, 0].reshape((batch_size, -1))
        server_comp_resource = batch['server'].x[:, 0].reshape((batch_size, -1))

        task_sche, power_sche, comp_sche = model(u2s_path_loss_feat, user_tasksize, server_comp_resource, u2s_index)
        loss = compute_loss_nn(task_sche, power_sche, comp_sche, user_tasksize, server_comp_resource, u2s_path_loss,
                               user_index, server_index)
        sum_loss += loss.item()
        length += 1
    mean_loss = sum_loss / length
    logger.info('Time step: {}, PCNet evaluate loss: {}'.format(step, mean_loss))
    return mean_loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def grid_search(nn_train_loader, global_model, user_models, nn_test_loader = None):
    tem = [10, 20, 30]
    beta = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    epoch = [1500, 3000]
    lr = [5e-4, 1e-4]
    lrf = [1e-5]
    tem = [10]
    beta = [0.5]
    epoch = [args.epoch]
    lr = [args.pcnet_lr]
    lrf = [args.lrf]
    # user_num = server_num * 3
    # input = server_num (10) * user_num (30) + server_num (10) + user_num (30)
    # output = server_num (10) * user_num (30)
    for a in tem:
        for b in beta:
            for c in epoch:
                for d in lr:
                    for e in lrf:
                        setup_seed(2024)
                        args.tem, args.beta, args.epoch, args.pcnet_lr, args.lrf = a, b, c, d, e
                        start = time.time()
                        NN_train(global_model, user_models, nn_train_loader)
                        # hgnn_model, train_loss, test_loss = HGNN_train(user_models, global_Model, gnn_train_layouts,
                        #                                                gnn_test_layouts)
                        end = time.time()
                        logger.info("training time: {}".format(end - start))
                        return None
                        # python .\flPretrain.py --min 20 --max 20 --epoch 100 --mode avg


if __name__ == '__main__':
    setup_seed(2024)
    logger.info(args)

    # setup_seed(2022)
    # Epochs = 20000
    Epochs = args.epoch
    global nn_lr
    train_num_layouts = args.train_num_layouts
    # train_num_layouts = 2048
    test_num_layouts = 512
    batch_size = 32
    batch_size = 8
    batch_size = 1
    eval_fre = 5
    save_fre = 50
    # multi_scales = 'FedAvg_SerEqUsr'
    multi_scales = f'{args.wandb_project}/NN_{args.mlp_wandb_name}'
    if multi_scales == 'small':
        min_server = 2
        max_server = 10
    elif multi_scales == 'medium':
        min_server = 15
        max_server = 25
    elif multi_scales == 'large':
        min_server = 25
        max_server = 35

    min_server = args.min
    max_server = args.max

    nn_train_server_nums = np.random.randint(min_server, max_server + 1, args.train_num_layouts)
    if not args.ratio_c:
        nn_train_user_nums = args.ratio * nn_train_server_nums
    else:
        nn_train_user_ratio = np.random.randint(args.ratio_cr[0],  args.ratio_cr[1] + 1, len(nn_train_server_nums))
        nn_train_user_nums = nn_train_server_nums * nn_train_user_ratio

    nn_eval_server_nums = nn_train_server_nums
    nn_eval_user_nums = nn_train_user_nums

    if 'trainOnlyOneGraph' in args.wandb_project:
        tidx = np.random.randint(0, len(nn_train_server_nums))
        nn_train_server_nums = [nn_train_server_nums[tidx]]
        nn_train_user_nums = [nn_train_user_nums[tidx]]

    args.server_list = nn_train_server_nums
    args.user_list = nn_train_user_nums
    args.eval_server_list = nn_eval_server_nums
    args.eval_user_list = nn_eval_user_nums
    print(nn_train_server_nums)
    print(nn_train_user_nums)

    nn_train_layouts = generate_layouts(nn_train_user_nums, nn_train_server_nums, args)
    nn_train_loader = DataLoader(nn_train_layouts, batch_size = batch_size, shuffle = False)
    nn_eval_layouts = generate_layouts(nn_eval_user_nums, nn_eval_server_nums, args)
    nn_eval_loader = DataLoader(nn_eval_layouts, batch_size = batch_size, shuffle = False)

    server = max(args.server_list)
    user = max(args.user_list)
    global_model = PCNet(server * user + server + user, args.hidden_dim, server * user, args).to(
        args.device)
    # 初始化用户端模型
    user_models = []
    for i in range(len(nn_train_user_nums)):
        temp = PCNet(server * user + server + user, args.hidden_dim, server * user, args).to(
            args.device)
        user_models.append(temp)

    # os.environ["WANDB_API_KEY"] = '67cca8dbf766a5a36297dff8d7388009b5ff2ed8'
    # os.environ["WANDB_MODE"] = "offline"
    if 'wyh' in os.getcwd():
        'HTTP_PROXY=http://127.0.0.1:7890;HTTPS_PROXY=http://127.0.0.1:7890'
        os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
        os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    grid_search(nn_train_loader, global_model, user_models)

    # from tkinter import messagebox
    # import tkinter as tk
    #
    # root = tk.Tk()
    # root.withdraw()
    # window = tk.Toplevel(root)
    # window.attributes('-topmost', True)
    # messagebox.showinfo('提示', 'mlp lognn 运行完成!')
    # window.destroy()


    # 23-12-20 round4 训练时 loss 到达过1.几
    # 2023-12-26\r1beta0.4tem10epoch1500lr0.0005lrf1e-05

    # wandb agent xdu_ai/FL_HGNN/nndtkfrn
    # wandb sweep --project FL_HGNN .\sweep.yaml
    '''
    syncing is set to `offline` in this directory.  
    wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
    wandb: NOTE: use wandb sync --clean to delete 1 synced runs from local directory.
    wandb: NOTE: use wandb sync --sync-all to sync 1 unsynced runs from local directory.
    '''
    # now  wandb agent xdu_ai/FL_HGNN/cvqs6yhw
    # HTTP_PROXY=http://127.0.0.1:7890;HTTPS_PROXY=http://127.0.0.1:7890
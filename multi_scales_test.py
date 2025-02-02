import os

import torch
from arguments import args
import numpy as np
from torch_geometric.loader import DataLoader
from layouts import generate_layouts
from off_loading_models import TaskLoad, PCNet
from tqdm import tqdm
import random
from time import time
import pandas as pd
from GA_exp import Evolution
from caculate_mean import cal_mean


def compute_loss_nn(task_allocation, power_allocation, comp_allocation, task_size, compute_resource, path_losses, user_index, server_index):
    
    # task_size : vector N
    # task_allocation: mat pop_num x 3*M*N
    # index: vector 3*M*N
    
    epsilon = 1e-9
    extre = 1e-20
    server_index_first = server_index.reshape((batch_size, -1))[0]
    user_index_first = user_index.reshape((batch_size, -1))[0]
    # user_index = edge_index[0]      # s2u中源节点的索引
    # server_index = edge_index[1]    # s2u中目标节点的索引
    
    # power_allocation = torch.clamp(power_allocation, 1e-5, 1)
    pw_ini = power_allocation * path_losses    # mat pop_num x M*N
    # pw = torch.clamp(pw, 1e-5, 1)
    
    # 将信道状态过小的设置为0
    mask_pw = torch.where(pw_ini<args.pw_threshold)
    
    pw = pw_ini.clone()

    pw[mask_pw] = 0

    comp_allocation_clone = comp_allocation.clone()
    comp_allocation_clone[mask_pw] = 0
    comp_allocation_normed = torch.zeros((comp_allocation_clone.shape[0], comp_allocation_clone.shape[1], server_index_first[-1]+1), device=args.device)
    comp_allocation_normed.scatter_(2, server_index_first.repeat((batch_size, 1)).unsqueeze(2), comp_allocation_clone.unsqueeze(2))
    comp_allocation_normed = comp_allocation_normed.sum(1)[:, server_index_first]
    comp_allocation_normed = torch.div(comp_allocation_clone, comp_allocation_normed+extre)

    task_allocation_clone = task_allocation.clone()
    task_allocation_clone[mask_pw] = 0
    task_allocation_normed = torch.zeros((task_allocation_clone.shape[0], task_allocation_clone.shape[1], user_index_first[-1]+1), device=args.device)
    task_allocation_normed.scatter_(2, user_index_first.repeat((batch_size, 1)).unsqueeze(2), task_allocation_clone.unsqueeze(2))
    task_allocation_normed = task_allocation_normed.sum(1)[:, user_index_first]
    task_allocation_normed = torch.div(task_allocation_clone, task_allocation_normed+extre)

    # 计算速率
    pw_list = torch.zeros((pw.shape[0], pw.shape[1], server_index_first[-1]+1), device=args.device)   # mat pop_num x MN x N
    pw_list.scatter_(2, server_index_first.repeat((batch_size, 1)).unsqueeze(2), pw.unsqueeze(2))
    pws_list = pw_list.sum(1)[:, server_index_first]  # mat pop_num x MN

    interference = pws_list-pw
    rate = torch.log2(1+torch.div(pw, interference+epsilon))
    # rate = args.band_width * torch.log2(1+torch.div(pw, interference+epsilon))


    task_size = task_size[:, user_index_first]       # M*N    
    # task_size = task_size[user_index]*args.tasksize_cof       # 重复采样映射到边中
    tasks = task_size * task_allocation_normed   # mat pop_num x M*N

    compute_resource = compute_resource[:, server_index_first]
    # compute_resource = compute_resource[server_index]*args.comp_cof       # 

    comp = compute_resource * comp_allocation_normed


    # offloading_time = torch.div(tasks, rate+extre) * (args.tasksize_cof/args.band_width)
    offloading_time = torch.div(tasks, rate+extre)

    # compute_time = torch.div(tasks, comp+extre) * (args.tasksize_cof*args.cons_factor/args.comp_cof)
    compute_time = torch.div(tasks, comp+extre)

    time_loss = offloading_time + compute_time      # pop_num x MN
    assert torch.isnan(time_loss).sum()==0


    time_loss_list = torch.zeros((time_loss.shape[0], time_loss.shape[1], user_index_first[-1]+1), device=args.device)
    time_loss_list.scatter_(2, user_index_first.repeat((batch_size, 1)).unsqueeze(2), time_loss.unsqueeze(2))
    time_loss_list = time_loss_list.sum(1)      # pop_num x MN

    return time_loss_list.mean()


def compute_loss(task_allocation, power_allocation, comp_allocation, compute_resource, path_losses, task_size, user_index, server_index):
    
    epsilon = 1e-9
    extre = 1e-20
    # user_index = edge_index[0]      # s2u中源节点的索引
    # server_index = edge_index[1]    # s2u中目标节点的索引
    
    pw_ini = power_allocation.squeeze() * path_losses.squeeze()
    # pw小于阈值的对应power设为0
    pw = pw_ini.clone()
    # mask_pw = torch.where(pw_ini<args.pw_threshold)
    # pw[mask_pw] = 0

    pw_user_list = torch.zeros((len(pw), user_index[-1]+1), device=args.device)
    # pw_user_ini_list = torch.zeros((len(pw), user_index[-1]+1), device=args.device)
    # pw_user_ini_list.scatter_(1, user_index.unsqueeze(1), pw_ini.unsqueeze(1))
    pw_user_list.scatter_(1, user_index.unsqueeze(1), pw_ini.unsqueeze(1))
    # 如果某一个user的发射功率均位于阈值以下
    pw_masked = pw_user_list.clone()
    pw_masked[torch.where(pw_masked < args.pw_threshold)] = 0
    invalid_index = torch.where(pw_masked.sum(0)==0)[0]   # 是否有对所有server都低于阈值的
    # assert len(invalid_index)==0
    max_pw_index = pw_user_list[:, invalid_index].argmax(0) # 对所有server的pw都低于阈值的user 取信号最强的server
    pw_masked[max_pw_index, invalid_index] = pw_user_list[max_pw_index, invalid_index]
    pw = pw_masked.sum(1)
    mask_pw = torch.where(pw==0)

    pw_list = torch.zeros((len(pw), server_index[-1]+1), device=args.device)
    pw_list.scatter_(1, server_index.unsqueeze(1), pw.unsqueeze(1))
    
    pws_list = pw_list.sum(0)[server_index]


    interference = pws_list-pw
    rate = torch.log2(1+torch.div(pw, interference+epsilon))

    task_allocation_clone = task_allocation.clone().squeeze() + 1e-8
    task_allocation_clone[mask_pw] = 0
    task_allocation_normed = torch.zeros((len(task_allocation_clone), user_index[-1]+1), device=args.device)
    task_allocation_normed.scatter_(1, user_index.unsqueeze(1), task_allocation_clone.unsqueeze(1))
    # assert len(torch.where(task_allocation_normed.sum(0)==0)[0]) == 0
    task_allocation_normed_2 = task_allocation_normed.sum(0)[user_index]
    task_allocation_final = torch.div(task_allocation_clone, task_allocation_normed_2+extre)
    # task_allocation_clone =  softmax(task_allocation_clone, user_index)

    task_size = task_size[user_index]
    # task_size = task_size[user_index]*args.tasksize_cof       # 重复采样映射到边中
    
    tasks = task_size * task_allocation_final

    comp_allocation_clone = comp_allocation.clone().squeeze()
    comp_allocation_clone[mask_pw] = 0
    comp_allocation_normed = torch.zeros((len(comp_allocation_clone), server_index[-1]+1), device=args.device)
    comp_allocation_normed.scatter_(1, server_index.unsqueeze(1), comp_allocation_clone.unsqueeze(1))
    comp_allocation_normed_2 = comp_allocation_normed.sum(0)[server_index]
    comp_allocation_final = torch.div(comp_allocation_clone, comp_allocation_normed_2+extre)
    # compute_resource = compute_resource[server_index]*args.comp_cof       # 

    compute_resource = compute_resource[server_index]
    comp = compute_resource * comp_allocation_final


    # offloading_time = torch.div(tasks, rate+extre) * (args.tasksize_cof/args.band_width)
    offloading_time = torch.div(tasks, rate+extre)

    # compute_time = torch.div(tasks, comp+extre) * (args.tasksize_cof*args.cons_factor/args.comp_cof)
    compute_time = torch.div(tasks, comp+extre)

    # compute_time = torch.clamp(compute_time, -1, 3000)
    # offloading_time = torch.clamp(offloading_time, -1, 3000)

    time_loss = offloading_time + compute_time
    assert torch.isnan(time_loss).sum()==0

    time_loss_list = torch.zeros((len(time_loss), user_index[-1]+1), device=args.device)
    time_loss_list.scatter_(1, user_index.unsqueeze(1), time_loss.unsqueeze(1))
    time_loss_list = time_loss_list.sum(0)

    return time_loss_list.mean()


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def run(sample_nums, bp, server_num_list, user_num_list):
    if net == 'NN':
        with open('mlp_test_round.txt', 'r') as fr:
            round = int(fr.read())
        round += 1
        with open('mlp_test_round.txt', 'w') as fw:
            fw.write(str(round))
    else:
        with open('test_round.txt', 'r') as fr:
            round = int(fr.read())
        round += 1
        with open('test_round.txt', 'w') as fw:
            fw.write(str(round))
    print(round, 'round', '---------------------------------------')
    # setup_seed(2023)
    hgnn_lr = 1e-5
    pcnet_lr = 1e-4
    train_steps = 300

    # 生成测试数据
    multi_scales = 'small'
    # if multi_scales == 'small':
    #     min_server = 2
    #     max_server = 30
    # elif multi_scales == 'medium':
    #     min_server = 15
    #     max_server = 20
    # elif multi_scales == 'large':
    #     min_server = 25
    #     max_server = 30

    layouts_list = []

    for num, user in zip(server_num_list, user_num_list):
        server_nums = np.ones(sample_nums, dtype = np.int8) * num
        # user_nums = server_nums * 3
        user_nums = np.ones(sample_nums, dtype = np.int8) * user
        layouts = generate_layouts(user_nums, server_nums, args)
        # layouts = generate_layouts(user, server_nums, args)
        layouts_list.append(layouts)

    # test with multi_scales

    # inference directly
    # loading pretrained_model

    # fine-tunning

    # GA iteration

    # PcNet inference directly

    # PcNet training

    # task offloading trainning process
    # loaded_hgnn = torch.load(bp, map_location=args.device)
    if net == 'NN':
        server = max(server_num_list)
        user = max(user_num_list)
        loaded_hgnn = PCNet(server * user + server + user, args.hidden_dim, server * user, args).to(
        args.device)
        loaded_hgnn.load_state_dict(torch.load(bp))
    else:
        loaded_hgnn = TaskLoad(args.num_layers, args.input_dim, args.hidden_dim, args.max_server_num,
                               args.alpha).to(args.device)
        loaded_hgnn.load_state_dict(torch.load(bp))

    index = 0
    loaded_hgnn.eval()
    excel_id = 0
    for layouts, server_num in zip(layouts_list, server_num_list):
        print('Now test server_num == {}'.format(server_num))
        # loaded_hgnn = torch.load('./TO_models/Pretrain/{}/2023-11-26/round2'.format(multi_scales)+'/{}'.format(best_model), map_location=args.device)
        # loaded_nn = torch.load('./TO_models/Pretrain/{}/pre'.format(multi_scales)+'/PcNet_2048_500.pt', map_location=args.device)
        # inference directly
        IDgnn_loss_list = []
        IDgnn_time_list = []
        IDNN_loss_list = []
        IDNN_time_list = []

        print(
            '######################################################Inference Directly Testing######################################################')
        for graph in tqdm(layouts):
            # inference HGNN directly
            u2s_index = graph['user', 'u2s', 'server'].edge_index
            user_index = u2s_index[0]
            server_index = u2s_index[1]
            u2s_path_loss = graph['user', 'u2s', 'server'].path_loss.squeeze().reshape((batch_size, -1))
            u2s_path_loss_feat = graph['user', 'u2s', 'server'].edge_attr.squeeze().reshape((batch_size, -1))
            user_tasksize = graph['user'].x[:, 0].reshape((batch_size, -1))
            server_comp_resource = graph['server'].x[:, 0].reshape((batch_size, -1))
            start = time()
            if net == 'NN':
                output_len = server_num_list[index] * user_num_list[index]
                IDGnn_task_allocation, IDGnn_power_allocation, IDGnn_comp_allocation = loaded_hgnn(
                    u2s_path_loss_feat, user_tasksize, server_comp_resource, u2s_index, output_len)
            else:
                IDGnn_task_allocation, IDGnn_power_allocation, IDGnn_comp_allocation = loaded_hgnn(graph.x_dict,
                                                                                               graph.edge_index_dict,
                                                                                               graph.edge_attr_dict)
            end = time()
            # user_index = graph['user', 'u2s', 'server'].edge_index[0]
            # server_index = graph['user', 'u2s', 'server'].edge_index[1]
            loss_batch = compute_loss(IDGnn_task_allocation, IDGnn_power_allocation, IDGnn_comp_allocation,
                                      graph['server'].x[:, 0], graph['user', 'u2s', 'server'].path_loss,
                                      graph['user'].x[:, 0], user_index, server_index)
            IDgnn_time_list.append(end - start)
            IDgnn_loss_list.append(loss_batch.item())
        index += 1
            # inference PcNet directly
            # u2s_index = graph['user', 'u2s', 'server'].edge_index
            # user_index = u2s_index[0]
            # server_index = u2s_index[1]
            # u2s_path_loss = graph['user', 'u2s', 'server'].path_loss.squeeze().reshape((batch_size, -1))
            # u2s_path_loss_feat = graph['user', 'u2s', 'server'].edge_attr.squeeze().reshape((batch_size, -1))
            # user_tasksize = graph['user'].x[:, 0].reshape((batch_size, -1))
            # server_comp_resource = graph['server'].x[:, 0].reshape((batch_size, -1))
            #
            # user_index_cliped = user_index[torch.where(server_index < min_server)]
            # path_losses_feat_cliped = u2s_path_loss_feat.squeeze()[torch.where(server_index < min_server)]
            # server_index_cliped = server_index[torch.where(server_index < min_server)]
            # server_index_cliped = server_index_cliped[torch.where(user_index_cliped < 3*min_server)]
            # user_index_cliped = user_index_cliped[torch.where(user_index_cliped < 3*min_server)]
            # path_losses_feat_cliped = path_losses_feat_cliped[torch.where(user_index_cliped < 3*min_server)].reshape((batch_size, -1))
            # start = time()
            # # 前min_server规模的方案
            # IDNN_task_sche, IDNN_power_sche, IDNN_comp_sche = loaded_nn(path_losses_feat_cliped, user_tasksize[:, :3*min_server], server_comp_resource[:, :min_server], [user_index_cliped, server_index_cliped])
            # end = time()
            # task_sche= torch.rand(u2s_path_loss_feat.shape, device=args.device)
            # power_sche = torch.rand(u2s_path_loss_feat.shape, device=args.device)
            # comp_sche = torch.rand(u2s_path_loss_feat.shape, device=args.device)
            # task_sche[:, :3*min_server**2] = IDNN_task_sche
            # power_sche[:, :3*min_server**2] = IDNN_power_sche
            # comp_sche[:, :3*min_server**2] = IDNN_comp_sche
            # IDNN_time_list.append(end-start)
            # loss = compute_loss_nn(task_sche, power_sche, comp_sche, user_tasksize, server_comp_resource, u2s_path_loss, user_index, server_index)
            # IDNN_loss_list.append(loss.item())

        ID_df = pd.DataFrame({
            'ID_HGNN_time': IDgnn_time_list,
            # 'ID_NN_time':IDNN_time_list,
            'ID_HGNN_loss': IDgnn_loss_list,
            # 'ID_NN_loss':IDNN_loss_list
        })

        # path = './data/multi_scales_test/{}/round{}'.format(multi_scales, round)
        path = r"E:\wyh\论文\UNIC\LOGNN\data\{}\{}\round{}".format(net, common_name, round)
        if not os.path.exists(path):
            os.makedirs(path)
        ID_df.to_excel(path + '/InferenceDirectly_{}_{}.xlsx'.format(server_num, excel_id))
        excel_id += 1

        # print('######################################################Fine_Tuning--Training Testing######################################################')
        # # for graph in layouts:
        # FTgnn_loss_list = []
        # FTnn_loss_list = []
        # FTgnn_time_list = []
        # FTnn_time_list = []
        # for graph in tqdm(layouts):    # graph为一个batch
        #     gnn_loss_idx = []
        #     hgnn_model = loaded_hgnn
        #     nn_model = PCNet((server_num**2)*3+server_num*4, args.hidden_dim, (server_num**2)*3, args.alpha).to(args.device)
        #     hgnn_optimizer = torch.optim.Adam(hgnn_model.parameters(), lr=hgnn_lr)
        #     nn_optimizer = torch.optim.Adam(nn_model.parameters(), lr=pcnet_lr)
        #     # hgnn fine-tuning
        #     gnn_time_idx= []

        #     for time_step in range(train_steps):
        #         start = time()
        #         task_allocation, power_allocation, comp_allocation = hgnn_model(graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict)
        #         edge_index = graph['user', 'u2s', 'server'].edge_index
        #         user_index = edge_index[0]
        #         server_index = edge_index[1]
        #         loss_batch = compute_loss(task_allocation, power_allocation, comp_allocation, graph['server'].x[:, 0], graph['user', 'u2s', 'server'].path_loss, graph['user'].x[:, 0], user_index, server_index)
        #         hgnn_optimizer.zero_grad()
        #         loss_batch.backward()
        #         hgnn_optimizer.step()
        #         end = time()
        #         gnn_time_idx.append(end-start)
        #         gnn_loss_idx.append(loss_batch.item())
        #     gnn_loss_idx = np.array(gnn_loss_idx)
        #     gnn_time_idx = np.array(gnn_time_idx)
        #     FTgnn_loss_list.append(gnn_loss_idx)
        #     FTgnn_time_list.append(gnn_time_idx)

        #     # pcNet fine-tunning
        #     u2s_index = graph['user', 'u2s', 'server'].edge_index
        #     user_index = u2s_index[0]
        #     server_index = u2s_index[1]
        #     u2s_path_loss = graph['user', 'u2s', 'server'].path_loss.squeeze().reshape((batch_size, -1))
        #     u2s_path_loss_feat = graph['user', 'u2s', 'server'].edge_attr.squeeze().reshape((batch_size, -1))
        #     user_tasksize = graph['user'].x[:, 0].reshape((batch_size, -1))
        #     server_comp_resource = graph['server'].x[:, 0].reshape((batch_size, -1))

        #     # 神经网络直接优化
        #     nn_loss_idx = []
        #     nn_time_idx = []
        #     for time_step in range(train_steps):
        #         start = time()
        #         task_sche, power_sche, comp_sche = nn_model(u2s_path_loss_feat, user_tasksize, server_comp_resource, u2s_index)
        #         loss = compute_loss_nn(task_sche, power_sche, comp_sche, user_tasksize, server_comp_resource, u2s_path_loss, user_index, server_index)
        #         nn_optimizer.zero_grad()
        #         loss.backward()
        #         nn_optimizer.step()
        #         end = time()
        #         nn_time_idx.append(end-start)
        #         nn_loss_idx.append(loss.item())
        #     nn_loss_idx = np.array(nn_loss_idx)
        #     nn_time_idx = np.array(nn_time_idx)
        #     FTnn_loss_list.append(nn_loss_idx)
        #     FTnn_time_list.append(nn_time_idx)

        # FTgnn_loss_list = np.vstack(FTgnn_loss_list)
        # FTgnn_loss_list = FTgnn_loss_list.mean(axis=0)     # N个样本求平均, 一个server_num

        # FTgnn_time_list = np.vstack(FTgnn_time_list)
        # FTgnn_time_list = FTgnn_time_list.mean(axis=0)

        # FTnn_loss_list = np.vstack(FTnn_loss_list)
        # FTnn_loss_list = FTnn_loss_list.mean(axis=0)

        # FTnn_time_list = np.vstack(FTnn_time_list)
        # FTnn_time_list = FTnn_time_list.mean(axis=0)

        # FT_df = pd.DataFrame({
        #                 'FT_HGNN_loss': FTgnn_loss_list,
        #                 'FT_NN_loss': FTnn_loss_list,
        #                 'FT_HGNN_time': FTgnn_time_list,
        #                 'FT_NN_time': FTnn_time_list
        # })
        # FT_df.to_excel('./data/multi_scales_test/'+multi_scales+'/FineTunning_{}.xlsx'.format(server_num))

        # GA 测试

        def GA_test():
            print(
                '######################################################GA Testing######################################################')
            pop_num = 200
            epochs = 300

            ga_loss_list = []
            ga_time_list = []
            for data in tqdm(layouts):
                compute_resource = data['server'].x[:, 0].squeeze()
                path_losses = data['user', 'u2s', 'server'].path_loss.squeeze()
                task_size = data['user'].x[:, 0].squeeze()
                edge_index = data['user', 'u2s', 'server'].edge_index
                ga = Evolution(server_num, server_num * 3, pop_num, epochs, compute_resource = compute_resource,
                               path_losses = path_losses, task_size = task_size, edge_index = edge_index)
                loss, ga_time = ga.evolution()
                ga_time_list.append(ga_time)
                ga_loss_list.append(loss.data.cpu().numpy())
            ga_loss_list = np.vstack(ga_loss_list)
            ga_loss_list = ga_loss_list.mean(axis = 0)
            ga_time_list = np.vstack(ga_time_list)
            ga_time_list = ga_time_list.mean(axis = 0)

            GA_df = pd.DataFrame({
                'GA_loss': ga_loss_list,
                'GA_time': ga_time_list
            })
            GA_df.to_excel('./data/multi_scales_test/' + multi_scales + '/GA_{}.xlsx'.format(server_num))

    path = r"E:\wyh\论文\UNIC\LOGNN\data\{}\{}\round{}".format(net, common_name, round)
    return round, cal_mean(path)


def cal_one(sample_nums, bp, server_num_list, user_num_list):
    run(sample_nums, bp, server_num_list, user_num_list)


if __name__=='__main__':
    setup_seed(2024)
    min_server = args.min
    max_server = args.max
    batch_size = 1
    # # net = 'NN'
    sample_nums = 3
    # bp = r"E:\wyh\论文\UNIC\LOGNN\TO_models\NN\NN_avg1-1_epo500\2024-4-19\r1beta0.5tem10epoch500lr0.001lrf0.001\long_time_no_save\NN_10_4.969347763061523_467.pt"
    # bp = r'E:\wyh\论文\UNIC\LOGNN\TO_models\NN\NN_avg1-1_isp5_ex1e-20\2024-4-18\r1beta0.5tem10epoch1000lr0.001lrf0.001\long_time_no_save\NN_10_4.9401277224222815_837.pt'
    # bp = r'E:\wyh\论文\UNIC\LOGNN\TO_models\NN\NN_prox1-1_epo500\2024-4-19\r1beta0.5tem10epoch500lr0.001lrf0.001\long_time_no_save\NN_10_4.971437668800354_456.pt'
    # bp = r'E:\wyh\论文\UNIC\LOGNN\TO_models\NN\NN_prox1-1_isp5_ex1e-20\2024-4-18\r1beta0.5tem10epoch1000lr0.001lrf0.001\long_time_no_save\NN_10_4.941692288716634_946.pt'
    # bp = r'E:\wyh\论文\UNIC\LOGNN\TO_models\NN\NN_avg1-1_epo500\2024-4-19\r1beta0.5tem10epoch500lr0.001lrf0.001\NN_10_35_4.992679802576701.pt'
    # bp = r'E:\wyh\论文\UNIC\LOGNN\TO_models\Pretrain\GNN_avg2-2_epo500\2024-4-19\r1beta0.5tem20epoch500lr0.005lrf0.001\HGNN_30_4.807545105616252_293.pt'
    # bp = r'E:\wyh\论文\UNIC\LOGNN\TO_models\NN\NN_avg2-2_epo500\2024-4-19\r1beta0.5tem10epoch500lr0.001lrf0.001\NN_10_248_14.87039779027303.pt'
    # bp = r'E:\wyh\论文\UNIC\LOGNN\TO_models\NN\NN_avg2-2_epo500\2024-4-19\r1beta0.5tem10epoch500lr0.001lrf0.001\long_time_no_save\NN_10_25.279561122258503_170.pt'
    # bp = r'E:\wyh\论文\UNIC\LOGNN\TO_models\NN\NN_avg2-2_epo500\2024-4-19\r1beta0.5tem10epoch500lr0.001lrf0.001\NN_10_500_10.429206498463948.pt'
    server_num_list = np.random.randint(min_server, max_server + 1, args.train_num_layouts)
    # if 'NN\\NN_' in bp:
    #     net = 'NN'
    # elif 'GNN' in bp:
    #     net = 'GNN'
    #
    # net_dict = {
    #     'NN': args.mlp_wandb_name,
    #     'GNN': args.wandb_name
    # }
    #
    # common_name = net_dict[net]
    if not args.ratio_c:
        user_num_list = server_num_list * args.ratio
    else:
        user_num_list = np.random.randint(args.ratio_cr[0],  args.ratio_cr[1] + 1, len(server_num_list))
    #
    # args.server_list = server_num_list
    # args.user_list = user_num_list
    # cal_one(sample_nums, bp, args.server_list, args.user_list)
    # exit()
    '''

    10 11.111545499165853
    11 8.221052169799805

    12 5.4908141454060875
    13 6.545669841766357
    14 5.809708245595297
    15 5.382261927922567
    16 3.8163300355275473
    17 4.30611404577891
    18 3.5237356662750243
    19 4.176857980092366
    20 3.5082505702972413
    mean 5.626576375238823
    '''
    # catalog = r'E:\wyh\论文\UNIC\LOGNN\TO_models\NN\NN_NN_avg4-1_epo500\2024-4-24\r1beta0.5tem10epoch500lr0.001lrf0.001'
    catalog = r'E:\wyh\论文\UNIC\LOGNN\TO_models\NN\NN_avg4-1_isp5_ex1e-20\2024-4-18\r1beta0.5tem10epoch1000lr0.001lrf0.001'
    files = os.listdir(catalog)
    best_mean = 65536
    best_mean_round = -1
    best_bp = ''
    for f in files:
        if os.path.isdir(os.path.join(catalog, f)):
            continue
        bp = os.path.join(catalog, f)
        if 'NN\\NN_' in bp:
            net = 'NN'
        elif 'GNN' in bp:
            net = 'GNN'

        net_dict = {
            'NN': args.mlp_wandb_name,
            'GNN': args.wandb_name
        }

        common_name = net_dict[net]
        rd, res = run(sample_nums, bp, server_num_list, user_num_list)
        if res < best_mean:
            best_mean = res
            best_mean_round = rd
            best_bp = bp
        print('------------------------------------')
    print(best_mean, best_mean_round)
    print(best_bp)

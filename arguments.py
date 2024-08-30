import os
import argparse
import logging
import sys
from ruamel.yaml import YAML



args = argparse.ArgumentParser()
#offloading 模型参数
args.add_argument('--learning_rate', default=1e-4)
args.add_argument('--critic_lr', default=5e-4)
args.add_argument('--input_dim', default=2)
args.add_argument('--state_dim', default=3)
args.add_argument('--action_dim', default=2)
args.add_argument('--hidden_dim', default=64)
args.add_argument('--alpha', default=0.2)
args.add_argument('--device', default='cuda:0')
args.add_argument('--load_pretrained', default=False)
args.add_argument('--num_layers', default=2)

# 实验环境参数
args.add_argument('--user_num', default=20)
args.add_argument('--server_num', default=5)
args.add_argument('--test_user_num', default=20)
args.add_argument('--test_server_num', default=5)
args.add_argument('--p_max', default=1)
args.add_argument('--pw_threshold', default=1e-6)
args.add_argument('--train_layouts', default=128)
args.add_argument('--test_layouts', default=64)
args.add_argument('--env_max_length', default=400)
args.add_argument('--server_height', default=20)
args.add_argument('--carrier_f_start', default=2.4e9)
args.add_argument('--carrier_f_end', default=2.4835e9)
args.add_argument('--signal_cof', default=4.11)
args.add_argument('--band_width', default=1e6)
args.add_argument('--batch_size', default=32)
args.add_argument('--max_server_num', default=15)
args.add_argument('--init_min_size', default=2)
args.add_argument('--init_max_size', default=8)

args.add_argument('--cons_factor', default=10)
args.add_argument('--init_min_comp', default=0.1)
args.add_argument('--init_max_comp', default=1)
args.add_argument('--comp_cof', default=1024**2)
args.add_argument('--tasksize_cof', default=1024*100)


args.add_argument('--multi_scales_train', default=False)

args.add_argument('--multi_scales_test', default=False)

args.add_argument('--single_scale_test', default=True)

args.add_argument('--comparison_hgnn', default=True)
args.add_argument('--comparison_pcnet', default=True)
args.add_argument('--comparison_pcnetCritic', default=False)


args.add_argument('--train_steps', default=600)
args.add_argument('--evaluate_steps', default=10)
args.add_argument('--save_steps', default=50)

# 我后续加的
args.add_argument('--model_name', type = str, default = 'FLHgnn')
# args.add_argument('--lrf', type = float, default = 0.0001)

args.add_argument('--max', type = int, default = 5)
args.add_argument('--min', type = int, default = 2)
args.add_argument('--ratio_c', type = int, default = 0, help = 'ratio change sign')
args.add_argument('--ratio_cr', default = [3, 4], help = 'change ratio range')

args.add_argument('--train_num_layouts', type = int, default = 30)
args.add_argument('--use_lrScheduler', type = bool, default = True)
args.add_argument('--tem', type = int, default = 20)
args.add_argument('--beta', type = float, default = 0.5)
args.add_argument('--max_save_interval', type = float, default = 50)
args.add_argument('--epoch', type = int, default = 500)
args.add_argument('--thre', type = float, default = 15)
args.add_argument('--extre', default = 1e-20, help = 'divisor minimal count')
args.add_argument('--ratio', type = int, default = 3, choices = [3, 5])
args.add_argument('--use_wandb', type = bool, default = False)
args.add_argument('--mode', type = str, default = 'normal')
args.add_argument('--inner_step', type = int, default = 5, choices = [1, 5])
args.add_argument('--lr', type = float, default = 5e-3)
args.add_argument('--pcnet_lr', type = float, default = 1e-3)
args.add_argument('--lrf', type = float, default = 1e-3)
args.add_argument('--flMode', type = str, default = 'avg', choices = ['avg', 'prox', 'kd'])
args.add_argument('--args_key', type = str, default = '4-1',
                  # choices = ["1-1", "2-1", "2-2", "3-1", "3-2", "4-1", '4-2', '4-3', '4-4', '4-5', '4-6']
                  )
args.add_argument('--prox_coef', type = float, default = 0.01)
# args.add_argument('--wandb_project', type = str, default = 'FL_LOGNN4')
# args.add_argument('--wandb_project', type = str, default = 'globalModelOnEachGraph')
args.add_argument('--wandb_project', type = str, default = 'LOGNN_RealProx')
# trainOnlyOneGraph
# fl lognn 3: 修改了 generate_layouts, 将 comp_source 和 path_loss 初始化范围改为 0.1-1, extre 改回 1e-20
# sed -i 's/\r//' a.sh 去除sh文件中的'\r'字符
t = args.parse_args()
with open('./pattern.yaml') as args_file:
    # args_key = "-".join([t.model_name, t.dataset, t.custom_key])
    args_key = t.args_key
    # args_key = "1-1"
    # args_key = "2-1"
    # args_key = "2-2"
    # args_key = "3-1"
    # args_key = "3-2"
    # args_key = "4-1"
    # args_key = "4-2"
    # args_key = "4-3"
    # args_key = "4-4"
    # args_key = "4-5"
    # args_key = "4-6"
    try:
        args.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
    except KeyError:
        raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")
# t = args.parse_args()
# if t.ratio_c:
#     common_name = \
#         f'{t.flMode}{args_key}_s{t.min}{t.max}_u{t.ratio_cr[0]}{t.ratio_cr[1]}s_isp{t.inner_step}_ex{t.extre}'
# else:
#     common_name = f'{t.flMode}{args_key}_s{t.min}{t.max}_u{t.ratio}s_isp{t.inner_step}_ex{t.extre}'
# common_name = f'{t.flMode}{args_key}_isp{t.inner_step}_ex{t.extre}'
common_name = f'{t.flMode}{args_key}_epo{t.epoch}'
args.add_argument('--wandb_name', type = str,
                  default = f'GNN_{common_name}')
args.add_argument('--mlp_wandb_name', type = str,
                  default = f'NN_{common_name}')
# 记录：第一次固定server为2，user/server = 3， isp = 5
# s25_su13_isp5: server 数量从 2 -> 5, server / user = 1 / 3, inner_step = 5
args = args.parse_args()

from datetime import datetime
args.log_name = '{}_{}.log'.format(args.model_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
if not os.path.exists('logs'):
    os.makedirs('logs')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))


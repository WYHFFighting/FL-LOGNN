# -*- coding: utf-8 -*-
import argparse
import logging
import os
import sys
from datetime import datetime


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tem', type = int, default = 20)
    parser.add_argument('--alpha', type = float, default = 0.5)
    parser.add_argument('--model_name', type = str, default = '')

    args = parser.parse_args()

    args.log_name = '{}_{}.log'.format(args.model_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))

    return args, logger
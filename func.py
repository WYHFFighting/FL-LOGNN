# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


def loss_fn_kd(outputs, labels, teacher_outputs, args):
    alpha = args.alpha
    T = args.tem
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim = 1),
                             F.softmax(teacher_outputs / T, dim = 1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1 - alpha)

    return KD_loss

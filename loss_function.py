import numpy as np
import torch
import copy
import sys

def loss_cal(output, batch_y, loss_func):
    accurate = 0
    # batch_y = batch_y.reshape(1)
    loss = loss_func(output, batch_y)
    pred_y = torch.max(output, 1)[1]
    pred_y = pred_y.numpy()
    batch_y = batch_y.numpy()
    if pred_y == batch_y:
        accurate = 1

    return loss, pred_y, accurate
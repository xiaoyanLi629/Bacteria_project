import numpy as np
import torch
import copy
import sys
from loss_function import loss_cal

def test_images(x, y, num, model, loss_func, device):
    """ Testing test data... """
    """ Return accuracy, prediction class, total loss"""
    x = x.to(device)
    x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
    output = model(x)
    output = output.to('cpu')
    y = y.reshape(1)
    loss = loss_func(output, y)
    loss.backward()
    return output, y


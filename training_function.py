import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import datetime
import time
from test_images import test_images
import torch.utils.data as Data


def train_func(model, loader, loss_func, optimizer, device):
    # torch.multiprocessing.freeze_support()
    for step, (batch_x, batch_y) in enumerate(loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)        
        output = model(batch_x)
        loss = loss_func(output, batch_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradientsd
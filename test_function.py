
import numpy as np
import torch
from loss_function import loss_cal
from test_images import test_images


def test_func(X, Y, model, loss_function, device):
    
    total_loss = 0
    train_accuracy = 0
    test_prediction = np.zeros(len(Y))

    for i in range(len(Y)):
        x = X[i]
        y = Y[i]
        
        output, batch_y = test_images(x, y, len(Y), model, loss_function, device)
        loss, pred_y, accurate = loss_cal(output, batch_y, loss_function)

        test_prediction[i] = pred_y
        total_loss = total_loss + loss
        train_accuracy = train_accuracy + accurate


    train_accuracy = train_accuracy/len(Y)*100

    return total_loss, train_accuracy, test_prediction

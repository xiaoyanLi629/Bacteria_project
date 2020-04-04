#This code is used to plot box plot of accuracy
#There are total 2000 iterations in training
#Only the last 1000 iterations produce a stable model

import pickle
import numpy as np
import matplotlib.pyplot as plt


pickle_in = open("test_accuracy.pickle","rb")
test_accuracy = pickle.load(pickle_in)
accuracy = test_accuracy[1000:]

average = np.average(accuracy)
std = np.std(accuracy)

max_acc = max(accuracy)
min_acc = min(accuracy)

print('Accuracy average:', average)
print('Accuracy standard deviation:', std)
print('Accuracy max:', max_acc)
print('Accuracy min:', min_acc)

pickle_in = open("test_prediction.pickle","rb")
test_prediction = pickle.load(pickle_in)
test_prediction = test_prediction.astype(int)
np.savetxt('test_prediction.csv', test_prediction)

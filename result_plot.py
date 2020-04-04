import pickle
import numpy as np
import matplotlib.pyplot as plt


# pickle_in = open("train_accuracy.pickle","rb")
# train_accuracy_list = pickle.load(pickle_in)
# pickle_in = open("test_accuracy.pickle","rb")
# test_accuracy_list = pickle.load(pickle_in)

# epoch_list = range(len(train_accuracy_list))

# train = plt.scatter(epoch_list, train_accuracy_list, s=5, c='b', alpha=0.3)
# test = plt.scatter(epoch_list, test_accuracy_list, s=5, c='r', alpha=0.3)
# plt.xlabel('Iteration')
# plt.ylabel('Accuracy')
# plt.title('CNN model train/test dataset accuracy vs model training iterations')
# plt.legend((train, test), ('Training data', 'Testing data'), scatterpoints=1, loc = 'upper left', fontsize=9)

# plt.savefig('model_accuracy.png')


pickle_in = open("X_train","rb")
X_train = pickle.load(pickle_in)
print(X_train.shape)

pickle_in = open("Y_train","rb")
Y_train = pickle.load(pickle_in)
print(Y_train.shape)

pickle_in = open("X_test","rb")
X_test = pickle.load(pickle_in)
print(X_test.shape)

pickle_in = open("Y_test","rb")
Y_test = pickle.load(pickle_in)
print(Y_test.shape)
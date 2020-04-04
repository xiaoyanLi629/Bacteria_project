import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import datetime
import time
from test_images import test_images
import pickle
import torch.utils.data as Data
from training_function import train_func
from loss_function import loss_cal
from test_function import test_func

# from loss import test

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=8,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=5),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(8, 16, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(5),  # output shape (32, 7, 7)
        )
        self.linear1 = nn.Linear(6400, 2048)
        self.linear_temp = nn.Linear(2048, 512)
        self.linear2 = nn.Linear(512, 32)
        self.out = nn.Linear(32, 4)
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear_temp(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        output = self.out(x)
        return output #, x  # return x for visualization


print('Current time', str(datetime.datetime.now()))
start = time.time()
model_name = 'Urine_CNN_model'

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train = pickle.load(open('X_train', "rb"))
Y_train = pickle.load(open('Y_train', "rb"))
X_test = pickle.load(open('X_test', "rb"))
Y_test = pickle.load(open('Y_test', "rb"))

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
Y_train = torch.from_numpy(Y_train).long()
Y_test = torch.from_numpy(Y_test).long()

basewidth = 500
EPOCH = 1000
LR = 0.001  # learning rate
stable_factor = 0  # control model stability

print('Epoch:', EPOCH)
print('Learning rate:', LR)

BATCH_SIZE = 96
torch_dataset_train = Data.TensorDataset(X_train, Y_train)
torch_dataset_test = Data.TensorDataset(X_test, Y_test)

loader_training = Data.DataLoader(
    dataset=torch_dataset_train,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)

loader_train_loss = Data.DataLoader(
    dataset=torch_dataset_train,      # torch TensorDataset format
    batch_size=1,      # mini batch size
    shuffle=False,               # random shuffle for training
    num_workers=0,
)

loader_test_loss = Data.DataLoader(
    dataset=torch_dataset_test,      # torch TensorDataset format
    batch_size=1,      # mini batch size
    shuffle=False,               # random ssses for loading data
)

cnn = CNN()
cnn.to(device)

# print(cnn)
print('Start training...')
x = list()
train_accuracy_list = []
test_accuracy_list = []
plt.ion()
epoch_list = []
train_accuracy_list = []
test_accuracy_list = []
accuracy_index = 0

for epoch in range(EPOCH):

    optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    train_func(cnn, loader_training, loss_func, optimizer, device)
    if epoch%100 == 0:
        LR = LR/2
    # print('epoch', epoch, 'Allocated memory:', torch.cuda.memory_allocated(device=None)/1024/1024, 'MB')

    if epoch%10 == 0:
        epoch_list.append(epoch)

        train_total_loss, train_accuracy, train_prediction = test_func(X_train, Y_train, cnn, loss_func, device)
        test_total_loss, test_accuracy, test_prediction = test_func(X_test, Y_test, cnn, loss_func, device)

        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)

        train = plt.scatter(epoch_list, train_accuracy_list, s=10, c='b', alpha=0.3)
        test = plt.scatter(epoch_list, test_accuracy_list, s=10, c='r', alpha=0.3)
        
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('CNN model train/test dataset accuracy vs model training iterations')
        plt.legend((train, test), ('Training data', 'Testing data'), scatterpoints=1, loc = 'upper left', fontsize=9)
        plt.show()
        plt.pause((0.01))

    print('Epoch: ', epoch, '| Train loss: %.8f' % train_total_loss, '| Train accuracy: %.8f' % train_accuracy, '%', '| Test loss: %.8f' % test_total_loss, '| Test accuracy: %.8f' % test_accuracy, '%')
    if test_accuracy >= 60:
        accuracy_index = accuracy_index + 1
    if accuracy_index >= 5:
        break

train = plt.scatter(epoch_list, train_accuracy_list, s=5, c='b', alpha=0.3)
test = plt.scatter(epoch_list, test_accuracy_list, s=5, c='r', alpha=0.3)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('CNN model train/test dataset accuracy vs model training iterations')
plt.legend((train, test), ('Training data', 'Testing data'), scatterpoints=1, loc = 'upper left', fontsize=9)

plt.savefig('model_accuracy.png')
torch.save(cnn, 'model_.pkl')
print('mode_accuracy figure has been saved')

pickle_out = open("train_accuracy.pickle","wb")
pickle.dump(train_accuracy_list, pickle_out)
pickle_out.close()
print('Train accuracy data has been saved')

pickle_out = open("test_accuracy.pickle","wb")
pickle.dump(test_accuracy_list, pickle_out)
pickle_out.close()
print('Test accuracy data has been saved')

pickle_out = open("train_prediction.pickle","wb")
pickle.dump(train_prediction, pickle_out)
pickle_out.close()
print('Train prediction result has been saved')

pickle_out = open("test_prediction.pickle","wb")
pickle.dump(test_prediction, pickle_out)
pickle_out.close()
print('Test prediction result has been saved')

done = time.time()
elapsed = (done - start)/60
print('Programming running time(mins):', elapsed)

os.system('spd-say "your program has finished"')


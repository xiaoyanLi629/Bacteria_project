import glob
import pickle
from PIL import Image
import PIL
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import os

class_count = 60
path, dirs, files = next(os.walk("Images/"))
file_count = len(files)

num_images = int(file_count)
foldername = 'Images' + '/'
image_format = '*.' + 'jpg'
basewidth = int(500)

image_data = np.zeros((num_images*4, 3,  basewidth, basewidth))

img_index = 0
for filename in sorted(glob.glob(foldername + image_format)):
    img = Image.open(filename)
    img = img.resize((500, 500))
    img_data = np.asarray(img, dtype='uint8')
    image_data[img_index, 0, :, :] = img_data[:, :, 0]
    image_data[img_index, 1, :, :] = img_data[:, :, 1]
    image_data[img_index, 2, :, :] = img_data[:, :, 2]
    img_index = img_index + 1

for m in range(3):
    for i in range(num_images):
        for j in range(3):
            image_data[num_images*(m+1)+i, j, :, :] = np.rot90(image_data[num_images*m+i, j, :, :], axes = (1, 0))

label = pd.read_csv('labels.csv', header=None)

x_train_temp = np.zeros((int(num_images*0.8)*4, 3, 500, 500))
x_test_temp = np.zeros((int(num_images*0.2)*4, 3, 500, 500))
y_train_temp = np.zeros((int(num_images*0.8)*4, 1))
y_test_temp = np.zeros((int(num_images*0.2)*4, 1))

for j in range(4):
    for i in range(4):
        X_train, X_test, y_train, y_test = train_test_split(image_data[i*class_count+j*num_images:(i+1)*class_count+j*num_images, :, :, :], label[i*class_count:(i+1)*class_count], test_size=0.2, random_state=42)
        
        x_train_temp[i * int(class_count * 0.8) + int(j * num_images * 0.8) : (i+1) * int(class_count * 0.8) + int(j * num_images * 0.8)] = X_train
        y_train_temp[i * int(class_count * 0.8) + int(j * num_images * 0.8) : (i+1) * int(class_count * 0.8) + int(j * num_images * 0.8)] = y_train
        x_test_temp[i * int(class_count * 0.2) + int(j * num_images * 0.2) : (i + 1) * int(class_count * 0.2) + int(j * num_images * 0.2)] = X_test
        y_test_temp[i * int(class_count * 0.2) + int(j * num_images * 0.2) : (i + 1) * int(class_count * 0.2) + int(j * num_images * 0.2)] = y_test

y_train_temp = y_train_temp.reshape(len(y_train_temp))
y_test_temp = y_test_temp.reshape(len(y_test_temp))

y_train_temp = y_train_temp.astype(int)
y_test_temp = y_test_temp.astype(int)



# print(x_train_temp.shape)
# print(x_train_temp)

# ave_file = save_file + '.pkl'
output = open('X_train', 'wb')
pickle.dump(x_train_temp, output, protocol=4)
output.close()

output = open('X_test', 'wb')
pickle.dump(x_test_temp, output, protocol=4)
output.close()

output = open('Y_train', 'wb')
pickle.dump(y_train_temp, output, protocol=4)
output.close()

output = open('Y_test', 'wb')
pickle.dump(y_test_temp, output, protocol=4)
output.close()


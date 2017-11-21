import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

K.set_image_dim_ordering('tf')
#path1 = '/home/faust/Documents/Python_Code/CNN_LP'
path1 = os.getcwd()
data_folder = 'data_dir'
data_path = path1 + '/' + data_folder
data_dir_list = os.listdir(data_path)

#Variables and constants
rows = 32
cols = 32
channel = 1
num_epoch = 10
num_classes = 36
folder_counter = -1
image_data = []
label_data = []
    
#Reading training data and formatting
for dataset in sorted(data_dir_list):
    img_list = os.listdir(data_path + '/' + dataset)
    folder_counter += 1
    print('Loaded images of folder '+ '{}'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img, 0)
        input_img_resize = cv2.resize(input_img,(rows,cols))
        image_data.append(input_img_resize)
        label_data.append(folder_counter)

label_data = np.array(label_data)
image_data = np.array(image_data)
image_data = image_data.astype('float32')
image_data /= 255
print('Dimensions of image_data: ')
print(image_data.shape)

image_data = np.expand_dims(image_data, axis=4)
print(image_data.shape)

#lables to one-hot
one_hot_labels = np_utils.to_categorical(label_data, num_classes)

#Creating training and testing datasets
data_train, data_test, labels_train, labels_test = train_test_split(image_data, one_hot_labels, test_size = 0.25, random_state =2)

input_shape = image_data[0].shape

model = Sequential()

#Input Layer
model.add(Convolution2D(32, 3, 3, border_mode = 'same', input_shape = input_shape))
model.add(Activation('relu'))

#Hidden Layers
model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#Output Layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

#Train the network
training = model.fit(
        data_train,
        labels_train,
        batch_size = 36,
        epochs = num_epoch,
        verbose = 1,
        validation_data = (data_test, labels_test),
        shuffle = True
        )

#Save the model
model.save('CNNModel.h5')

'''
#Visualize
train_loss = training.history['loss']
val_loss = training.history['val_loss']
train_acc = training.history['acc']
val_acc = training.history['val_acc']
xc = range(num_epoch)

plt.figure(1,figsize = (7,5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])
'''

score = model.evaluate(data_test, labels_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = data_test[0:1]
print (test_image.shape)

print(model.predict(test_image))
print(model.predict_classes(test_image))
print(labels_test[0:1])












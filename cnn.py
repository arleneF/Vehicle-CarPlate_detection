from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
from keras import backend as K
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import os
import tensorflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
K.set_image_dim_ordering('th')

os.chdir("/home/faust/Documents/Python_Code/keras_LP")

m,n = 32, 32

path1 = "input"
path2 = "data_dir"

classes = os.listdir(path2)
classes.sort()
x = []
y = []

for dir in classes:
    #print(dir)
    imgfiles = os.listdir(path2+'/'+dir)
    for img in imgfiles:
        im = Image.open(path2 + '/' + dir + '/' + img)
        im = im.convert('1')
        #imrs = im.resize((m,n))
        imrs = img_to_array(im)/255
       # imrs = imrs.transpose(2,0,1)
       # imrs = imrs.reshape(3,m,n)
        x.append(imrs)
        y.append(dir)

#print(x)
#print(y)

x = np.array(x)
y = np.array(y)

batch_size = 36
num_classes = len(classes)
num_epoch = 50
num_filters = 32
num_pool = 2
num_conv = 3

#Create testing data (one third of data set)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 4)

uniques, id_train = np.unique(y_train, return_inverse = True)
Y_train = np_utils.to_categorical(id_train, num_classes)
uniques, id_test = np.unique(y_test, return_inverse = True)
Y_test = np_utils.to_categorical(id_test, num_classes)

#Create model
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = 3, strides = 3, padding = 'same', input_shape = x_train.shape[1:]))

#Convolution2D deprecated
#model.add(Convolution2D(num_filters, num_conv, num_conv, padding = 'same', input_shape = x_train.shape[1:]))

model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (num_pool, num_pool)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])

#Print model summary
model.summary()

#Train network
model.fit(x_train, Y_train, batch_size = batch_size, epochs = num_epoch, verbose = 1, validation_data = (x_test, Y_test))

#Save model
model.save('model.h5')

#To Do:
#Implement character recognition  in seperate file


'''
files = os.listdir(path1)
files.sort()
img = files[0]
im = Image.open(path1 + '/' + img)
imrs = im.resize((m,n))
imrs = imrs.convert('1')
imrs = img_to_array(imrs)/255
imrs = imrs.transpose(2,0,1)
imrs = imrs.reshape(1, m, n)


x = []
x.append(imrs)
x = np.array(x)
predictions = model.predict(x)
'''



'''
********************************
License Plate Recogniction Program
By Karamveer Dhillon
November 20, 2017

Plate detection and character segmentation adapted from code from:
https://blog.devcenter.co/developing-a-license-plate-recognition-system-with-machine-learning-in-python-787833569ccd.

CNN Model from:
https://github.com/anujshah1003/own_data_cnn_implementation_keras/blob/master/custom_data_cnn.py

Data set for training CNN from:
http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

********************************
'''



from keras.models import load_model
from keras.utils import np_utils
from keras.applications.resnet50 import decode_predictions
import numpy as np
import cv2
import os
import character_detection
import matplotlib.pyplot as pyplt
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.io import imread

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Loading the previously created model
trained_model = load_model('CNNModel.h5')

#setting up path to folder containing saved chars
path = os.getcwd()
input_folder = 'input_dir'
input_path = path + '/' + input_folder
input_images_list = sorted(os.listdir(input_path))

rows, cols = 32, 32
channel = 1
base_name = 'writtenImage.jpg'
imCounter = 0


print("Modes: 1 to read from car image saved as 'test_car5.jpg' or 2 to read from character images saved in folder input_dir\n")
print('Currently, the program will only predict character images saved in folder. However, the code will still show plate detection and segmentation work for the test car image.\n')
prog_mode = input('Please enter Mode: ')
print('\n')

#-----# From character_detection (regions)
if prog_mode == '1':
    temparray = []
    #preprocessing and saving in an array
    for each_character in character_detection.licenseChars: 
        imCounter += 1
        fileName = str(imCounter) + base_name
        im = cv2.resize(each_character, (rows, cols))
        row, col = im.shape[:2]
        #Create a border of some white pixels around the segmented characters
        bottom = im[row-2:row, 0:col]
        mean = cv2.mean(bottom)[0]
        border = 10
        input_img = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value = 0)
        input_img = cv2.resize(input_img, (rows,cols))
        #Turning images to black and white
        otsu = threshold_otsu(input_img)
        input_img = input_img > otsu
        #changing type to uint8, in order to invert image from white on black to black on white
        input_img = input_img.astype('uint8')
        input_img = cv2.bitwise_not(input_img)
        print('Shape of cv2.resize input_img: ',input_img.shape)
        #Saving Images to folder - Bug: saved images are either all white or all black
        path ='/home/faust/Documents/Python_Code/CNN_LP/saved_dir/'
        cv2.imwrite(os.path.join(path, fileName), input_img)
        temparray.append(input_img)
        fig, (ax1) = pyplt.subplots(1)
        ax1.imshow(input_img, cmap="gray")

    pyplt.show()
    
    #list to numpy array -> expanding dimensions
    temparray = np.array(temparray)
    print('Shape of np array temparray: ', temparray.shape)
    temparray = temparray.astype('float32')
    temparray /= 255

    temparray = np.expand_dims(temparray, axis=4)
    print('shape of expanded temparray: ', temparray.shape)

#-----# From Folder
elif prog_mode == '2':

    input_data = []
    #read images and preproccess
    for img in input_images_list:
        input_img = cv2.imread(input_path + '/' + img, 0)
        print(input_img.shape)
        input_img_resize = cv2.resize(input_img,(rows,cols))
        print(input_img_resize.shape)
        input_data.append(input_img_resize)
        print('Loaded ', img)
        #Input_data is list, no shape at this point
    
    #list to numpy array. Cast as float, divide by 255 and expand dimensions
    input_data = np.array(input_data)
    input_data = input_data.astype('float32')
    input_data /= 255
    input_data = np.expand_dims(input_data, axis=4)
    #print('shape of input_data after expand dims',input_data.shape)
    # [3, 32, 32, 1], first is number of images, size, size, channels



# ** Prediction precentages - Unused for now **
#predictions = trained_model.predict(input_data)
#print('Predicted:', decode_predictions(predictions, top=3[0]))
# round predictions
#rounded = [round(x[0]) for x in predictions]
#print(rounded)
# ** ------------------- **

else:
    print('Invalid mode.')
    exit()

#Predicting depending on selected mode
if prog_mode == '1':
    y_pred = trained_model.predict_classes(temparray, 1, verbose=0)
if prog_mode == '2':
    y_pred = trained_model.predict_classes(input_data, 1, verbose=0)
#print(y_pred.shape)



#Switcher dict used to switch from int representation of classes to class names
switcher = {
                0: '0',
                1: '1',
                2: '2',
                3: '3',
                4: '4',
                5: '5',
                6: '6',
                7: '7',
                8: '8',
                9: '9',
                10: 'A',
                11: 'B',
                12: 'C',
                13: 'D',
                14: 'E',
                15: 'F',
                16: 'G',
                17: 'H',
                18: 'I',
                19: 'J',
                20: 'K',
                21: 'L',
                22: 'M',
                23: 'N',
                24: 'O',
                25: 'P',
                26: 'Q',
                27: 'R',
                28: 'S',
                29: 'T',
                30: 'U',
                31: 'V',
                32: 'W',
                33: 'X',
                34: 'Y',
                35: 'Z'}

translated = [switcher[num] for num in y_pred]
print(y_pred)
print('Predicted License Plate Number: ',translated)



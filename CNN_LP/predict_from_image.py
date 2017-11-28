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
trained_model = load_model('CNNModel.h5')

path = os.getcwd()
input_folder = 'input_dir'
input_path = path + '/' + input_folder
input_images_list = os.listdir(input_path)

rows, cols = 32, 32
channel = 1
base_name = 'writtenImage.jpg'
imCounter = 0


#-----# From character_detection (regions)

temparray = []

for each_character in character_detection.licenseChars: 
    imCounter += 1
    fileName = str(imCounter) + base_name
    
    im = cv2.resize(each_character, (rows, cols))
    row, col = im.shape[:2]
    bottom = im[row-2:row, 0:col]
    mean = cv2.mean(bottom)[0]
    
    border = 10
    input_img = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value = 0)
    
    #input_img = each_character
    input_img = cv2.resize(input_img, (rows,cols))
    otsu = threshold_otsu(input_img)
    input_img = input_img > otsu
    input_img = input_img.astype('uint8')
    input_img = cv2.bitwise_not(input_img)
    print('Shape of cv2.resize input_img: ',input_img.shape)
    path ='/home/faust/Documents/Python_Code/CNN_LP/saved_dir/'
    #cv2.imwrite(os.path.join(path, fileName), cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(path, fileName), input_img)
    temparray.append(input_img)
    fig, (ax1) = pyplt.subplots(1)
    ax1.imshow(input_img, cmap="gray")

pyplt.show()

temparray = np.array(temparray)
#np.bitwise_not(temparray)
#np.invert(temparray)
print('Shape of np array temparray: ', temparray.shape)
temparray = temparray.astype('float32')
temparray /= 255
#ax1.imshow(temparray[0], cmap = 'gray')
#pyplt.show()

temparray = np.expand_dims(temparray, axis=4)
print('shape of expanded temparray: ', temparray.shape)
#predictions = trained_model.predict(temparray)

'''
#----# From folder
input_data = []

for img in input_images_list:
    input_img = cv2.imread(input_path + '/' + img, 0)
    print(input_img.shape)
    input_img_resize = cv2.resize(input_img,(rows,cols))
    print(input_img_resize.shape)
    input_data.append(input_img_resize)
    print('Loaded ', img)
    #Input_data is list, no shape at this point

input_data = np.array(input_data)
input_data = input_data.astype('float32')
input_data /= 255
print('shape of input_data after np: ',input_data.shape)

input_data = np.expand_dims(input_data, axis=4)
print('shape of input_data after expand dims',input_data.shape)
print('This is the shape that the model should predict!!')
# [3, 32, 32, 1], first is number of images, size, size, channels



# ** Unused code for now **
#predictions = trained_model.predict(input_data)
#print('Predicted:', decode_predictions(predictions, top=3[0]))
# round predictions
#rounded = [round(x[0]) for x in predictions]
#print(rounded)
# ** ------------------- **
'''

y_pred = trained_model.predict_classes(temparray, 1, verbose=0)
#y_pred = trained_model.predict_classes(input_data, 1, verbose=0)
#print(y_pred.shape)
# ** ** ** ** ** ** ** ** **


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



'''
classification_result = []
for each_character in character_detection.licenseChars:
    each_character = each_character.reshape(1, -1);
    print(each_character.shape)
    result = trained_model.predict(each_character)

classification_result.append(result)
'''



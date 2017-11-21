from keras.models import load_model
from keras.utils import np_utils
from keras.applications.resnet50 import decode_predictions
import numpy as np
import cv2
import os



trained_model = load_model('CNNModel.h5')

path = os.getcwd()
input_folder = 'input_dir'
input_path = path + '/' + input_folder
input_images_list = os.listdir(input_path)

rows, cols = 32, 32
channel = 1

input_data = []

for img in input_images_list:
    input_img = cv2.imread(input_path + '/' + img, 0)
    input_img_resize = cv2.resize(input_img,(rows,cols))
    input_data.append(input_img_resize)
    print('Loaded ', img)

input_data = np.array(input_data)
input_data = input_data.astype('float32')
input_data /= 255
#print(input_data.shape)

input_data = np.expand_dims(input_data, axis=4)
#input_data = np.expand_dims(input_data, axis=3)
#input_data = np.expand_dims(input_data, axis=0)

#print(input_data.shape)

predictions = trained_model.predict(input_data)
#print(predictions)
#print('Predicted:', decode_predictions(predictions, top=3[0]))

# round predictions
#rounded = [round(x[0]) for x in predictions]
#print(rounded)
y_pred = trained_model.predict_classes(input_data, 1, verbose=0)
print(y_pred)
#print(y_pred.shape)

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
#print ''.join(translated)
print('Predicted License Plate Number: ',translated)







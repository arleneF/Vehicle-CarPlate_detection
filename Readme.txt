License Plate Location, Plate Segmentation and Character Recognition
By Karamveer Dhillon



This program is still a work in progress. As of now, the plate locator may locate more than one region as a possible plate. These regions are saved in an array, therefore when using your own input image, you may get something like a headlight as your located plate. You may change the array's index manually in plate_detection.py if you wish to use your own image.

The demo image, "test_car5.jpg" works well with index = 2.

Although the character detector works, there is a bug when sending it to the prediciton software. Therefore, currently the program does not support direct character recognition from the segmented license plate characters.

Instead, this demo contains a directory of the processed license plate characters of "test_car5.jpg". These images are the output of character_detection.py.



Using the program:

In the command line, run "Python3 predict_from_image.py"

There are two modes available when running the main program, predict_from_image.py.

Mode 1 is to detect directly from a car image located within the project folder. As highlighted above, thise feature does not currently work as intended. You will, however, see the output steps of all the components, including plate detection and character segmentation. However, the predicted characters will either be all I's or H's.

Mode 2 is for predicting the characters in a specific folder. For this demo, input_dir contains segmented characters from "test_car5.jpg". The output will be the predicted values using the pretrained model.

In order to try different image folders, simply replace the images in the input_dir directory with your own. There are additional images located within input_dir1, 2 and 3.

Summary of files:
	cnn.py - The keras/TensorFlow model of the neural network that was trained
	CNNModel.h5 - the saved CNN model
	plate_detection.py - Code that finds the plate within the car image
	character_detection.py - Code that finds and segments the individual characters within license plate
	predict_from_image.py - The prediciton software that predicts characters. This is the main file.
	test_car1.jpg to test_car7.jpg - test images of vehicles
	input_dir - input directory of segmented characters
	input_dir1 to input_dir3 - additional segmented characters that may be used for testing
	data_dir - directory containing the images used for training the CNN


Thank you :)






Reference Code Used:


Plate detection and character segmentation adapted from:
https://github.com/anujshah1003/own_data_cnn_implementation_keras/blob/master/custom_data_cnn.py

CNN Model adapted from:
https://github.com/anujshah1003/own_data_cnn_implementation_keras/blob/master/custom_data_cnn.py

Data set:
http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/


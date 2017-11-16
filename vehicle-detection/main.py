
import matplotlib.image as mpimg
from yolo_pipeline import *
from calibration import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from calibration import load_calibration
from copy import copy




def pipeline_yolo(img):
    input_scale = 1
    # load the calibration
    calib_file = 'calibration_pickle.p'
    mtx, dist = load_calibration(calib_file)
    # resize the input image according to scale
    img_undist_ = cv2.undistort(img, mtx, dist, None, mtx)
    img_undist = cv2.resize(img_undist_, (0,0), fx=1/input_scale, fy=1/input_scale)
    output = vehicle_detection_yolo(img_undist)

    return output



if __name__ == "__main__":


        filename = 'examples/car1.jpg'
        image = mpimg.imread(filename)
        image = cv2.resize(image,(1280,720))

        # Yolo pipeline
        yolo_result = pipeline_yolo(image)
        plt.figure()
        plt.imshow(yolo_result)
        plt.title('yolo pipeline', fontsize=30)
        # plt.savefig('examples/123.jpg')
        plt.show()

import numpy as np
import tensorflow as tf
from keras import backend as K
import cv2
from timeit import default_timer as timer
import time
import matplotlib.pyplot as plt
from visualizations import *

# --------Define the model--------- #
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU

class yolo_tf:
	# ---------Initialization-------- #
	w_img = 1280
	h_img = 720
	weights_file = 'weights/YOLO_small.ckpt'
	alpha = 0.1
	threshold = 0.3
	iou_threshold = 0.5
	result_list = None
	classes =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
                "sheep", "sofa", "train","tvmonitor"]

	def __init__(self):
		#tf.reset_default_graph()
		self.build_networks()

	def build_networks(self):
		print("Building YOLO_small graph...")
		#build a classifier
		# self.x=tf.placeholder(tf.float32,shape=(448,448,3))
		self.x=tf.placeholder(tf.float32,shape=(448,448,3))

		model=Sequential()
		#!!!!!!!one possible to change to make it work:
		#input_shape should link to x
		#and maybe try without activation = 'linear'
		model.add(Conv2D(64,(7,7),strides=2,input_shape=(448,448,3),activation='linear',padding='same')) #64 filters, each filter's size is (7*7)
		model.add(LeakyReLU(alpha=.1))
		model.add(MaxPooling2D(pool_size=(2,2),strides=2))

		model.add(Conv2D(192,(3,3),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(MaxPooling2D(pool_size=(2,2),strides=2))

		model.add(Conv2D(128,(1,1),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(Conv2D(256,(3,3),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(Conv2D(256,(1,1),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(Conv2D(512,(3,3),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(MaxPooling2D(pool_size=(2,2),strides=2))

		model.add(Conv2D(256,(1,1),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(Conv2D(512,(3,3),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(Conv2D(256,(1,1),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(Conv2D(512,(3,3),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(Conv2D(256,(1,1),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(Conv2D(512,(3,3),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(Conv2D(256,(1,1),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(Conv2D(512,(3,3),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(Conv2D(512,(1,1),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(Conv2D(1024,(3,3),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(MaxPooling2D(pool_size=(2,2),strides=2))

		model.add(Conv2D(512,(1,1),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(Conv2D(1024,(3,3),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(Conv2D(512,(1,1),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(Conv2D(1024,(3,3),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(Conv2D(1024,(3,3),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(Conv2D(1024,(3,3),strides=2,activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(Conv2D(1024,(3,3),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))
		model.add(Conv2D(1024,(3,3),activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=.1))

		model.add(Flatten())
		model.add(Dense(512)) #fully connected layer
		model.add(Dense(4096)) #fully connected layer
		model.add(Dense(1470)) #fully connected layer
		#!!!!!!!one possible to change to make it work:
		# yuan dai ma li de linear=true

		#model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
		#model.summary()

		sess=tf.Session() # creating a Tensorflow session
		#K.set_session(sess) # registering it with keras, this means that keras will use the session we registered to initialize all variables that it creates internally
		#initialize all variables
		init_op=tf.global_variables_initializer()
		sess.run(init_op)

		saver = tf.train.Saver()
		saver.restore(sess, self.weights_file)

		#use keras layers to speed up the model definition process
		#preds=model(x) #put placeholder in to defined model
		print("Loading complete!")

def detect_from_cvmat(yolo,img):
	#image.shape return (height, width, channel), but here we delibrately ignore channel variable
    yolo.h_img,yolo.w_img,_ = img.shape

    img_resized = cv2.resize(img, (448, 448))
    img_resized_np = np.asarray( img_resized ) #convert input to an array
    inputs = np.zeros((1,448,448,3),dtype='float32')
    inputs[0] = (img_resized_np/255.0)*2.0-1.0
    in_dict = {yolo.x: inputs}
    net_output = yolo.sess.run(yolo.model,feed_dict=in_dict)
    result = interpret_output(yolo, net_output[0])
    yolo.result_list = result


def detect_from_file(yolo,filename):
    detect_from_cvmat(yolo, filename)


def interpret_output(yolo,output):
    probs = np.zeros((7,7,2,20))
    class_probs = np.reshape(output[0:980],(7,7,20))
    scales = np.reshape(output[980:1078],(7,7,2))
    boxes = np.reshape(output[1078:],(7,7,2,4))
    offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))

    boxes[:,:,:,0] += offset
    boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
    boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
    boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
    boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])

    boxes[:,:,:,0] *= yolo.w_img
    boxes[:,:,:,1] *= yolo.h_img
    boxes[:,:,:,2] *= yolo.w_img
    boxes[:,:,:,3] *= yolo.h_img

    for i in range(2):
        for j in range(20):
            probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])

    filter_mat_probs = np.array(probs>=yolo.threshold,dtype='bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)
    boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]

    argsort = np.array(np.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort]
    probs_filtered = probs_filtered[argsort]
    classes_num_filtered = classes_num_filtered[argsort]

    for i in range(len(boxes_filtered)):
        if probs_filtered[i] == 0 : continue
        for j in range(i+1,len(boxes_filtered)):
            if iou(boxes_filtered[i],boxes_filtered[j]) > yolo.iou_threshold :
                probs_filtered[j] = 0.0

    filter_iou = np.array(probs_filtered>0.0,dtype='bool')
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]

    result = []
    for i in range(len(boxes_filtered)):
        result.append([yolo.classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

    return result


def draw_results(img, yolo, fps,counter):
    img_cp = img.copy()
    results = yolo.result_list

    # draw the highlighted background
    img_cp = draw_background_highlight(img_cp, yolo.w_img)

    window_list = []
    for i in range(len(results)):
        x = int(results[i][1])
        y = int(results[i][2])
        w = int(results[i][3])//2
        h = int(results[i][4])//2
        cv2.rectangle(img_cp,(x-w,y-h),(x+w,y+h),(0,0,255),4)
        cv2.rectangle(img_cp,(x-w,y-h-20),(x+w,y-h),(125,125,125),-1)
        # cv2.putText(img_cp,results[i][0] + ' : %.2f' % results[i][5],(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
        cv2.putText(img_cp,results[i][0],(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
        if results[i][0] == "car" or results[i][0] == "bus":
            window_list.append(((x-w,y-h),(x+w,y+h)))

    # draw vehicle thumbnails
    draw_thumbnails(img_cp, img, window_list,counter)
    return img_cp

def iou(box1,box2):
    tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
    lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
    if tb < 0 or lr < 0 : intersection = 0
    else : intersection =  tb*lr
    return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

yolo = yolo_tf()

def vehicle_detection_yolo(image,counter):
    # set the timer
    start = timer()
    detect_from_file(yolo, image)

    # compute frame per second
    fps = 1.0 / (timer() - start)
    # draw visualization on frame
    yolo_result = draw_results(image, yolo, fps,counter)

    return yolo_result

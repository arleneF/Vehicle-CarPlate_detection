import matplotlib.image as mpimg
from yolo_pipeline import *
import glob

from copy import copy

if __name__ == "__main__":
    counter = 0
    #for filename in glob.glob('C:/Users/Ricky/Desktop/cars_test/*.jpg'): #assuming jpg
    filename = 'C:/Users/Ricky/Desktop/cars_test/00033.jpg'
    image = mpimg.imread(filename)
    image = cv2.resize(image,(1280,720))
    print ('Counter:%d\n'% counter)
    print (image.shape)
    plt.figure()
    plt.imshow(image)
    # Yolo pipeline
    yolo_result = vehicle_detection_yolo(image,counter)
    #plt.figure()
    #plt.imshow(yolo_result)
    #plt.title('yolo pipeline', fontsize=30)
    plt.savefig('croppedImg/%d.jpg'%counter)
    #plt.show()
    counter = counter+1

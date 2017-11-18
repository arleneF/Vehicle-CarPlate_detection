from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as pyplt


#read image to 2d array, and set values as greyscale
carPic = imread("test_car1.jpg", as_grey=True)
#print(carPic.shape)
#print(carPic)


#Use Otsu's method to create a black&white image
#Display both greyscale and new Black&white image
#fig, (ax1, ax2) = pyplt.subplots(1,2)
#ax1.imshow(carPic, cmap="gray")
otsuLevel = threshold_otsu(carPic)
carPicBW = carPic > otsuLevel
#ax2.imshow(carPicBW, cmap="gray")
#pyplt.show()




#-------------------------------------------------------------------------#
#
# Character detection and segmentation using CCA
# This is a work in progress
#-------------------------------------------------------------------------#

from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as pyplt
import numpy as np
import plate_detection


#Depending on input image, the possiblePlates index may need to be manually changed
#For most images, index 0 will work best. For the demo (test_car5.jpg), index 2 is required

licensePlate = np.invert(plate_detection.possiblePlates[2])
#licensePlate = plate_detection.possiblePlates[0]
#licensePlate = np.array(licensePlate)
labelledPlate = measure.label(licensePlate)

fig, ax1 = pyplt.subplots(1)
ax1.imshow(licensePlate, cmap="gray")
pyplt.show()

#min and max dimensions for LP characters
charDimensions = (0.35*licensePlate.shape[0], 0.60*licensePlate.shape[0], 0.05*licensePlate.shape[1], 0.15*licensePlate.shape[1])
minCharHeight, maxCharHeight, minCharWidth, maxCharWidth = charDimensions

licenseChars = []
count = 0
colArray = []

#Iterate through labelled regions, and append the ones containing characters to an array
for regions in regionprops(labelledPlate):
    charY0, charX0, charY1, charX1 = regions.bbox
    charHeight = charY1 - charY0
    charWidth = charX1 - charX0

    if charHeight > minCharHeight and charHeight < maxCharHeight and charWidth > minCharWidth and charWidth < maxCharWidth:
        charRegion = licensePlate[charY0:charY1, charX0:charX1]

        regionBorder = patches.Rectangle((charX0-3, charY0-3), (charX1 - charX0) + 3, (charY1-charY0)+ 3, edgecolor = "red", linewidth = 2, fill = False)

        ax1.add_patch(regionBorder)

        resizedChar = resize(charRegion, (32,32))
        licenseChars.append(resizedChar)

        colArray.append(charX0)
pyplt.show()






    


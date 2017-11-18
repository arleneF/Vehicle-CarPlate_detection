#-------------------------------------------------------------------------#
#
# Character detection and segmentation using CCA
#
#-------------------------------------------------------------------------#

from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as pyplt
import numpy as np
import plate_detection

licensePlate = np.invert(plate_detection.possiblePlates[0])
labelledPlate = measure.label(licensePlate)

fig, ax1 = pyplt.subplots(1)
ax1.imshow(licensePlate, cmap="gray")
#pyplt.show()

#min and max dimensions for LP characters
charDimensions = (0.35*licensePlate.shape[0], 0.60*licensePlate.shape[0], 0.05*licensePlate.shape[1], 0.15*licensePlate.shape[1])
minCharHeight, maxCharHeight, minCharWidth, maxCharWidth = charDimensions

licenseChars = []
count = 0
colArray = []

for regions in regionprops(labelledPlate):
    charY0, charX0, charY1, charX1 = regions.bbox
    charHeight = charY1 - charY0
    charWidth = charX1 - charX0

    if charHeight > minCharHeight and charHeight < maxCharHeight and charWidth > minCharWidth and charWidth < maxCharWidth:
        roi = licensePlate[charY0:charY1, charX0:charX1]

        regionBorder = patches.Rectangle((charX0, charY0), charX1 - charX0, charY1-charY0, edgecolor = "red", linewidth = 2, fill = False)

        ax1.add_patch(regionBorder)

        resizedChar = resize(roi, (20,20))
        licenseChars.append(resizedChar)

        colArray.append(charX0)
print(colArray)
pyplt.show()

    


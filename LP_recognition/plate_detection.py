#--------------------------------------------------------------------------------#
#Using Connected Component Analysis (CCA) to detect connected regions
#
#
#---------------------------------------------------------------------------------#

from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as pyplt
import matplotlib.patches as patches
import binary_image

#use measure.label to group pixels together and store in labeledPic

labeledPic = measure.label(binary_image.carPicBW)

plateDimensions = (0.08*labeledPic.shape[0], 0.2*labeledPic.shape[0], 0.15*labeledPic.shape[1], 0.4*labeledPic.shape[1])
minHeight, maxHeight, minWidth, maxWidth = plateDimensions
possiblePlateLocation = []
possiblePlates = []

fig, (ax1) = pyplt.subplots(1)
ax1.imshow(binary_image.carPic, cmap="gray");

#iterate through regions in labledPic, check if region too small
#labledPic contains list of regions, regionprops method returns list of region's properties
#patches.Rectangle used to draw

imageSize = labeledPic.size
for region in regionprops(labeledPic):
    #default value is 50, 500 works well for most images
    if region.area < 0.001*imageSize: 
        continue
        
    minY, minX, maxY, maxX = region.bbox
    regionHeight = maxY - minY
    regionWidth = maxX - minX
    regionRatio = regionHeight/regionWidth

#Original check (only works with certain palte-to-image ratios)
    #if regionHeight >= minHeight and regionHeight <= maxHeight and regionWidth >= minWidth and regionWidth <= maxWidth and regionWidth > regionHeight:
 #       possiblePlates.append(binary_image.carPicBW[minY:maxY, minX:maxX])
  #      possiblePlateLocation.append((minY, minX, maxY, maxX))




#Check ratios only (small regins may be boxed)
    if regionRatio>= 0.40 and regionRatio <= 0.50:
        possiblePlates.append(binary_image.carPicBW[minY:maxY, minX:maxX])
        possiblePlateLocation.append((minY, minX, maxY, maxX))

#Check ratios and min/max values (size of plate wrt image may cause issues)
   # if regionHeight >= minHeight and regionHeight <= maxHeight and regionWidth >= minWidth and regionWidth <= maxWidth and regionRatio >= 0.40 and regionRatio <= 0.5:
    #    possiblePlates.append(binary_image.carPicBW[minY:maxY, minX:maxX])
     #   possiblePlateLocation.append((minY, minX, maxY, maxX))



        regionBox = patches.Rectangle((minX, minY), maxX-minX, maxY-minY, edgecolor = "red", linewidth = 2, fill = False)
       # ax1.add_patch(regionBox)

#print(len(possiblePlates))
#pyplt.show()



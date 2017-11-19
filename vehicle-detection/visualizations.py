import cv2
import numpy as np

def draw_thumbnails(img_cp, img, window_list ,counter):
    for i, bbox in enumerate(window_list):
        thumbnail = img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
        cv2.imwrite('croppedImg/sample%d_%d.jpg'%(counter,i),cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR))
        #the following is the extra small car projection , not necessary so deleted
        #vehicle_thumb = cv2.resize(thumbnail, dsize=(thumb_w, thumb_h))
        #start_x = 300 + (i+1) * off_x + i * thumb_w
        #img_cp[off_y + 30:off_y + thumb_h + 30, start_x:start_x + thumb_w, :] = vehicle_thumb

def draw_background_highlight(image, w):
    #outline the rect shape on the detected car
    mask = cv2.rectangle(np.copy(image), (0, 0), (w, 155), (0, 0, 0), thickness=cv2.FILLED)
    return image

import cv2
import numpy as np
import cv2


def draw_speed(img_cp, fps, w):

    fps_info = "{0:4.1f} fps".format(fps)
    cv2.putText(img_cp, 'Speed', (w - 120,37), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)
    cv2.putText(img_cp, fps_info, (w - 130,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 1, cv2.LINE_AA)
    cv2.line(img_cp,(w-160,0),(w-160,155),(255,0,0),5)


def draw_thumbnails(img_cp, img, window_list, thumb_w=100, thumb_h=80, off_x=30, off_y=30):

    cv2.putText(img_cp, 'Detected viehicles', (400,37), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)
    for i, bbox in enumerate(window_list):
        thumbnail = img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
        cv2.imwrite('croppedImg/sample.jpg',thumbnail)
        vehicle_thumb = cv2.resize(thumbnail, dsize=(thumb_w, thumb_h))
        start_x = 300 + (i+1) * off_x + i * thumb_w
        img_cp[off_y + 30:off_y + thumb_h + 30, start_x:start_x + thumb_w, :] = vehicle_thumb


def draw_background_highlight(image, w):

    mask = cv2.rectangle(np.copy(image), (0, 0), (w, 155), (0, 0, 0), thickness=cv2.FILLED)

    return image

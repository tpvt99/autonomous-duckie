import numpy as np
import cv2

# def select_rgb_white_yellow(image): 
#     # white color mask
#     lower = np.uint8([200, 200, 200])
#     upper = np.uint8([255, 255, 255])
#     white_mask = cv2.inRange(image, lower, upper)
#     # yellow color mask
#     lower = np.uint8([190, 190,   0])
#     upper = np.uint8([255, 255, 255])
#     yellow_mask = cv2.inRange(image, lower, upper)
#     # combine the mask
#     mask = cv2.bitwise_or(white_mask, yellow_mask)
#     masked = cv2.bitwise_and(image, image, mask = mask)
#     return masked

# def select_rgb_white_yellow(image): 
#     # white color mask
#     lower = np.uint8([200, 200, 200])
#     upper = np.uint8([255, 255, 255])
#     white_mask = cv2.inRange(image, lower, upper)
#     # yellow color mask
#     lower = np.uint8([150, 150,   0])
#     upper = np.uint8([255, 255, 255])
#     yellow_mask = cv2.inRange(image, lower, upper)
#     # combine the mask
#     mask = cv2.bitwise_or(white_mask, yellow_mask)
#     masked = cv2.bitwise_and(image, image, mask = mask)
#     return masked

def select_rgb_white_yellow(image): 
    # white color mask
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([150, 150,   0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # red color mask with red road lines
    lower = np.array([0,0,0])
    upper = np.array([255,80,90])
    # red color mask without red road lines
    # lower = np.array([0,0,0])
    # upper = np.array([255,255,0])
    red_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    mask = cv2.bitwise_or(mask, red_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked
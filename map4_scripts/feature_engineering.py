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

def get_dist_to_stop(env, cur_pos):
    dist_to_stop = 1000.0

    for obj in env.unwrapped.objects:
        if obj.kind == "sign_stop":
            dist_to_stop = min(dist_to_stop, ((cur_pos[0] - obj.pos[0]) ** 2 + (cur_pos[2] - obj.pos[2]) ** 2) ** 0.5)

    return dist_to_stop

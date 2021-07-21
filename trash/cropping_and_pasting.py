import cv2
import numpy as np
import fnmatch
import os
import random
from PIL import Image
import time

start = time.time()


def alpha_cutter(img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img[:, :, 3] = mask
    cv2.imwrite("test_alpha.png", img)
    # cv2.imshow('trans', img)
    # cv2.waitKey(0)
    return img


def negative_filter(img):
    return cv2.bitwise_not(img)


def get_object_area_and_mask(img):
    # bilateral filter
    image = cv2.bilateralFilter(img, 75, 75, 75)
    # canny
    edges = cv2.Canny(image, 100, 255, L2gradient=True)
    # dilate mask
    dilate_range = 25
    kernel = np.ones((dilate_range, dilate_range), 'uint8')
    th = cv2.dilate(edges, kernel, iterations=1)

    # # show image
    # cv2.imshow("mask", th)
    # cv2.imshow("image", img)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (125, 125, 0), 3)

    x_list = []
    y_list = []
    for x, y, w, h in map(cv2.boundingRect, contours):
        x_list += [x, x + w]
        y_list += [y, y + h]
    if x_list and y_list:
        x0, y0, x1, y1 = min(x_list), min(y_list), max(x_list), max(y_list)
        # msk blur
        blur_range = 20
        th = cv2.blur(th, (blur_range, blur_range))
        return th, x0, y0, x1, y1
    else:
        cv2.imshow('broken image', img)
        cv2.waitKey(0)


def file_finder(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def crop_and_convert(img):
    # cropping image
    mask, x0, y0, x1, y1 = get_object_area_and_mask(img)
    img = alpha_cutter(img, mask)
    img = img[y0:y1, x0:x1]
    return img


def overlay_transparent(background_img, img_to_overlay_t, x, y):
    """
    @brief      Overlays a transparant PNG onto another image using CV2

    @param      background_img    The background image
    @param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
    @param      x                 x location to place the top-left corner of our overlay
    @param      y                 y location to place the top-left corner of our overlay
    @param      overlay_size      The size to scale our overlay to (tuple), no scaling if None

    @return     Background image with overlay on top
    """

    bg_img = background_img.copy()
    # Extract the alpha mask of the RGBA image, convert to RGB
    b, g, r, a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b, g, r))
    print(overlay_color.shape)
    # Optional, apply some simple filtering to the mask to remove edge noise
    mask = a

    h, w, _ = overlay_color.shape
    x, y = int(x - (float(w) / 2.0)), int((y - float(h) / 2.0))
    roi = bg_img[y:y + h, x:x + w]

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)
    # Update the original image with our new ROI
    bg_img[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)
    return bg_img


def paste_image(background, foreground, coords):
    background = background.convert('RGBA')
    foreground = foreground.convert('RGBA')
    background.paste(foreground, coords, mask=foreground)
    return background


def data_frame_generator(data_dir, background_path):
    background = cv2.imread(background_path)
    raw_obj = os.listdir(data_dir)
    obj_names = [obj for obj in raw_obj if '.csv' not in obj]
    new_x = 0

    for obj_name in obj_names:
        img_name = random.choice(os.listdir(f"{data_dir}/{obj_name}"))
        foreground = crop_and_convert(cv2.imread(f'{data_dir}/{obj_name}/{img_name}')[0:300, 50:300])

        new_y = random.randint(0, background.shape[0] - foreground.shape[0])
        if new_x > background.shape[1] + foreground.shape[1]:
            continue
        # background = paste_image(background, foreground, (new_x, new_y))
        background = overlay_transparent(background, foreground, new_x, new_y)
        new_x = new_x + foreground.shape[1] + random.randint(0, 60)
    # background.show()
    cv2.imshow('result', background)
    cv2.waitKey(0)


data_frame_generator('./object_dataset_2021-07-02', 'background.png')
print(time.time() - start)

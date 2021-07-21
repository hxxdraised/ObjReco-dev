import os
import cv2
import time
import random
from fnmatch import fnmatch
import numpy as np
import csv

# ==========================================================
BACKGROUND_PATH = 'background.png'
SOURCE_PATH_PREFIX = 'object_dataset_2021-07-09'
NEW_PATH_PREFIX = 'auto_object_dataset_2021-07-09'
# ds settings
FRAMES = 1000
TRAIN_PART = 8
VALIDATE_PART = 1
TEST_PART = 1
# resize
BACK_RESIZE_TO = 600  # height of background image


# RESIZE_TO = 400  # height of result image
# ===========================================================


def alpha_cutter(img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img[:, :, 3] = mask
    cv2.imwrite("test_alpha.png", img)
    return img


def negative_filter(img):
    return cv2.bitwise_not(img)


def get_object_area_and_mask(img):
    # bilateral filter
    image = cv2.bilateralFilter(img, 75, 75, 75)
    # canny
    edges = cv2.Canny(image, 100, 255, L2gradient=True)
    # dilate mask
    dilate_range = 22
    kernel = np.ones((dilate_range, dilate_range), 'uint8')
    th = cv2.dilate(edges, kernel, iterations=1)

    # show image
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
        blur_range = 10
        th = cv2.blur(th, (blur_range, blur_range))
        # cv2.imshow('mask', th)
        # cv2.waitKey(0)
        return th, x0, y0, x1, y1
    else:
        raise Exception("Broken image")


def file_finder(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        result += [os.path.join(root, name) for name in files if fnmatch(name, pattern)]
    return result


def crop_and_convert(img):
    # cropping image
    mask, x0, y0, x1, y1 = get_object_area_and_mask(img)
    img = alpha_cutter(img, mask)
    img = img[y0:y1, x0:x1]
    return img


def overlay_transparent(background, foreground, x, y):
    # if background does not have alpha channel -- add
    if foreground.shape[-1] == 3:
        background = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA)
        background[:, :, 3] = 255

    alpha = foreground[:, :, 3] / 255
    h, w, _ = foreground.shape
    height = slice(y, h + y)
    width = slice(x, w + x)
    for channel in range(3):
        background[height, width, channel] = (1 - alpha) * background[height, width, channel] + alpha * foreground[:, :,
                                                                                                        channel]
    # background = cv2.rectangle(background, (x, y), (x + w, y + h), (255, 0, 0), 3)
    return background


def get_ds_type(cur_frame):
    if cur_frame % (TRAIN_PART + VALIDATE_PART + TEST_PART) < TRAIN_PART:
        return "TRAIN"
    elif TRAIN_PART <= cur_frame % (TRAIN_PART + VALIDATE_PART + TEST_PART) < TRAIN_PART + VALIDATE_PART:
        return "VALIDATE"
    return "TEST"


def generate_frame(data_dir, background, annotation_writer, frame_path, ds_type):
    start = time.time()

    height, width, _ = background.shape
    ratio = height / width
    background = cv2.resize(background, (BACK_RESIZE_TO, int(BACK_RESIZE_TO * ratio)))
    # background shape
    bg_height, bg_width, _ = background.shape

    raw_obj = os.listdir(data_dir)
    obj_names = [obj for obj in raw_obj if not obj.endswith('.csv')]
    prepared_files = {obj_name: os.listdir(f"{data_dir}/{obj_name}") for obj_name in obj_names}

    new_x = 0
    while new_x < bg_width:
        obj_name = random.choice(obj_names)
        img_name = random.choice(prepared_files[obj_name])
        foreground = cv2.imread(f'{data_dir}/{obj_name}/{img_name}')[0:300, 50:300]
        try:
            foreground = crop_and_convert(foreground)
        except Exception as e:
            print(f"{e}: {obj_name}")
            continue
        # foreground shape
        fg_height, fg_width, _ = foreground.shape

        new_y = random.randint(0, bg_height - fg_height)
        if new_x + fg_width > bg_width:
            break

        background = overlay_transparent(background, foreground, new_x, new_y)
        annotation_writer.writerow([ds_type, frame_path, obj_name,
                                    round(new_x / bg_width, 3), round(new_y / bg_height, 3), None, None,
                                    round((new_x + fg_width) / bg_width, 3), round((new_y + fg_height) / bg_height, 3),
                                    None, None])
        new_x += fg_width + random.randint(0, 80)

    print(time.time() - start)
    return background


if __name__ == "__main__":
    background_img = cv2.imread('background.jpg')
    os.makedirs(f'{NEW_PATH_PREFIX}', exist_ok=True)
    with open(f'{NEW_PATH_PREFIX}/csv_annot.csv', 'a') as csv_annotation:
        annot_writer = csv.writer(csv_annotation, delimiter=',')
        for f_counter in range(FRAMES):
            frame_path = f"{NEW_PATH_PREFIX}/{f_counter}.jpg"
            frame = generate_frame(SOURCE_PATH_PREFIX, background_img, annot_writer, frame_path, get_ds_type(f_counter))
            cv2.imwrite(frame_path, frame)
            print(f"frame {f_counter}: Done!")

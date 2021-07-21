import cv2
import numpy as np
import fnmatch
import os
import csv

# ==========================================================
LINK_PREFIX = 'auto_object_dataset_2021-07-09/'
TRAIN_PART = 8
VALIDATE_PART = 1
TEST_PART = 1
RESIZE_TO = 300  # height of resized image


# ===========================================================

def file_finder(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def get_ds_type(cur_frame):
    if cur_frame % (TRAIN_PART + VALIDATE_PART + TEST_PART) < TRAIN_PART:
        return "TRAIN"
    elif TRAIN_PART <= cur_frame % (TRAIN_PART + VALIDATE_PART + TEST_PART) < TRAIN_PART + VALIDATE_PART:
        return "VALIDATE"
    return "TEST"


def get_object_area(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((25, 25), 'uint8')
    image = cv2.dilate(th, kernel, iterations=1)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_list = []
    y_list = []
    for x, y, w, h in map(cv2.boundingRect, contours):
        x_list += [x, x + w]
        y_list += [y, y + h]
    x0, y0, x1, y1 = min(x_list), min(y_list), max(x_list), max(y_list)

    return x0, y0, x1, y1


os.makedirs(f'./{LINK_PREFIX[:-1]}', exist_ok=True)
annotations = open(f'./{LINK_PREFIX[:-1]}/csv_annot.csv', 'a')
annot_write = csv.writer(annotations, delimiter=',')

path = './'
video_links = file_finder('*.mp4', path)
for link in video_links:
    cap = cv2.VideoCapture(link)
    video_name = str(link.split('/')[-1][:-4])
    print(video_name)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for cur_frame in range(frames):
        # get frame by frame
        ret, frame = cap.read()
        if ret:
            height, width, _ = frame.shape
            ratio = width / height
            img = cv2.resize(frame, (int(RESIZE_TO * ratio), RESIZE_TO))
        else:
            print(f"{cur_frame} is Broken Image")
            continue
        #
        os.makedirs(f'./{LINK_PREFIX[:-1]}/{video_name}/', exist_ok=True)
        cv2.imwrite(f'./{LINK_PREFIX[:-1]}/{video_name}/{cur_frame}.jpeg', img)

        ds_type = get_ds_type(cur_frame)
        x_s, y_s, x_e, y_e = get_object_area(img)
        # print(x_s, y_s, x_e, y_e)
        annot_write.writerow([ds_type,
                              LINK_PREFIX + video_name + '/' + str(cur_frame) + '.jpeg', video_name,
                              round(x_s / RESIZE_TO / ratio, 3), round(y_s / RESIZE_TO, 3), None, None,
                              round(x_e / RESIZE_TO / ratio, 3), round(y_e / RESIZE_TO, 3), None, None])

        if cur_frame % 50 == 0:
            print(f"{cur_frame}/{frames}")

annotations.close()

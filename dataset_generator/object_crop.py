import cv2
import numpy as np

CANNY_THRESH_1 = 50
CANNY_THRESH_2 = 160
VERBOSE = True


# noinspection SpellCheckingInspection
def get_borders_and_mask(img, process_type):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if process_type == "solid":
        mask = process_solid(img_grey)
    elif process_type == "transparent":
        mask = process_transparent(img_grey)
    else:
        raise ValueError("Incorrect process type argument")

    # corner indexes: top left corner & low right corner
    corn_indx = get_borders(mask)

    return mask, corn_indx


def process_solid(img):
    w, h = img.shape
    img = cv2.resize(img, (h // 3, w // 3)) * 2
    img = cv2.GaussianBlur(img, (3, 3), 0)

    canny_output = cv2.Canny(img, 50, 160)

    # ________improving edge_quality________
    kernel_dil = np.ones((7, 7), np.uint8)
    dilation_noise = cv2.dilate(canny_output, kernel_dil, iterations=1)

    kernel_cl = np.ones((10, 10), np.uint8)
    open_noise = cv2.morphologyEx(dilation_noise, cv2.MORPH_OPEN, kernel_cl)

    kernel_dil = np.ones((7, 7), np.uint8)
    dilation = cv2.dilate(open_noise, kernel_dil, iterations=1)

    kernel_cl = np.ones((7, 7), np.uint8)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_cl)

    kernel_op = np.ones((7, 7), np.uint8)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_op)

    canny_better = opening
    # ______________________________________

    # find contours
    contours, _ = cv2.findContours(canny_better, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # use image contours to make a mask
    mask = np.zeros((canny_output.shape[0], canny_output.shape[1]), dtype=np.uint8)
    for i in range(len(contours)):
        cv2.fillPoly(mask, pts=[contours[i]], color=255)

    # blur mask to avoid rough edges
    blurred_mask = cv2.GaussianBlur(mask, (11, 11), 0)

    if VERBOSE:
        img_horizontal = np.concatenate((img, canny_output, dilation, closing, blurred_mask), axis=1)
        cv2.imshow("Mask", img_horizontal)
        cv2.waitKey(0)

    blurred_mask_big = cv2.resize(blurred_mask, (h, w))

    return blurred_mask_big


def process_transparent(img):
    w, h = img.shape
    img = cv2.resize(img, (h // 2, w // 2)) * 3
    img = cv2.GaussianBlur(img, (3, 3), 0)

    canny_output = cv2.Canny(img, 50, 160)

    # ________improving edge_quality________
    kernel_dil = np.ones((7, 7), np.uint8)
    dilation_noise = cv2.dilate(canny_output, kernel_dil, iterations=1)

    kernel_op = np.ones((7, 7), np.uint8)
    open_noise = cv2.morphologyEx(dilation_noise, cv2.MORPH_OPEN, kernel_op)

    kernel_dil = np.ones((15, 15), np.uint8)
    dilation = cv2.dilate(open_noise, kernel_dil, iterations=1)

    kernel_cl = np.ones((7, 7), np.uint8)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_cl)

    kernel_op = np.ones((7, 7), np.uint8)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_op)

    canny_better = opening
    # ______________________________________

    # find contours
    contours, _ = cv2.findContours(canny_better, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # use image contours to make a mask
    mask = np.zeros((canny_output.shape[0], canny_output.shape[1]), dtype=np.uint8)
    for i in range(len(contours)):
        cv2.fillPoly(mask, pts=[contours[i]], color=255)

    # blur mask to avoid rough edges
    blurred_mask = cv2.GaussianBlur(mask, (11, 11), 0)

    if VERBOSE:
        img_horizontal = np.concatenate((img, canny_output, dilation, closing, blurred_mask), axis=1)
        cv2.imshow("Mask", img_horizontal)
        cv2.waitKey(0)

    blurred_mask_big = cv2.resize(blurred_mask, (h, w))

    return blurred_mask_big


def get_borders(mask):
    try:
        where = np.array(np.where(mask))
        x0, y0 = np.amin(where, axis=1)
        x1, y1 = np.amax(where, axis=1)
        return [[x0, y0], [x1, y1]]
    except ValueError:
        print("Incorrect mask format")


def apply_mask(frame, mask):
    # apply mask on alpha channel to leave only object
    img_alpha = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    img_alpha[:, :, 3] = mask

    return img_alpha


def get_object_from_frame(frame_path, process_type):
    frame = cv2.imread(frame_path)
    mask, corn_indx = get_borders_and_mask(frame, process_type)

    object_only_img = apply_mask(frame, mask)

    # crops frame to leave only object
    x0, y0, x1, y1 = corn_indx[0][0], corn_indx[0][1], corn_indx[1][0], corn_indx[1][1]
    cropped_object = object_only_img[x0:x1, y0:y1]

    return cropped_object


if __name__ == "__main__":
    frame_path = "object_frames/transparent/borjomi/Picture 513.jpg"
    only_object = get_object_from_frame(frame_path, process_type="transparent")
    cv2.imwrite("test_obj.png", only_object)

import cv2
import numpy as np



def alpha_cutter(img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img[:, :, 3] = mask
    print(img)
    cv2.imwrite("test_alpha.png", img)
    cv2.imshow('trans', img)
    cv2.waitKey(0)
    return img


def get_object_mask(img):
    # bilateral filter
    image = cv2.bilateralFilter(img, 75, 75, 75)
    # canny
    edges = cv2.Canny(image, 100, 255, L2gradient=True)
    # dilate mask
    dilate_range = 25
    kernel = np.ones((dilate_range, dilate_range), 'uint8')
    th = cv2.dilate(edges, kernel, iterations=1)
    # msk blur
    blur_range = 20
    th = cv2.blur(th, (blur_range, blur_range))
    # show image

    cv2.imshow("image", img)
    cv2.imshow("mask", th)
    return th


img = cv2.imread("object_dataset_2021-07-02/borjomi_can/123.jpeg")
mask = get_object_mask(img)
alpha_cutter(img, mask)

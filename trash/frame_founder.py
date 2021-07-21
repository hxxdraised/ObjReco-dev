# import cv2
# import numpy as np
#
#
# def get_object_area(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
#     ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     kernel = np.ones((25, 25), 'uint8')
#     image = cv2.dilate(th, kernel, iterations=1)
#     contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(img, contours, -1, (125, 125, 0), 3)
#
#     x_list = []
#     y_list = []
#     for x, y, w, h in map(cv2.boundingRect, contours):
#         x_list += [x, x + w]
#         y_list += [y, y + h]
#     x0, y0, x1, y1 = min(x_list), min(y_list), max(x_list), max(y_list)
#     # -- Show image --
#     cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 2)
#     cv2.imshow('dilated', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     return (x0, y0), (x1, y1)
#
#
#
# img = cv2.imread('./actimel_bottle/600.jpeg')
# print(get_object_area(img))
#
import os
print(os.getcwd())
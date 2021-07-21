import cv2

img = cv2.imread('./test.jpg')
img = cv2.resize(img, (300, 300))
cv2.imwrite("resized.jpg", img)

import imutils
import cv2
import time
import numpy as np

cap = cv2.VideoCapture('/dev/video2')
# cap_real = cv2.VideoCapture('/dev/video0')
saliency = None


def negative(img):
    image = cv2.bitwise_not(img)
    return image


# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

frame_average = []
for i in np.arange(0, 2):
    ret, frame = cap.read()
    average = frame.mean(axis=0).mean(axis=0)[0]
    print(f'{average} average')
    frame_average.append(average)

light_frame = frame_average.index(max(frame_average))
print(light_frame)

frame_counter = 0
while True:
    ret, frame = cap.read()
    # _, frame_real = cap_real.read()
    average = frame.mean(axis=0).mean(axis=0)[0]
    if average < 10:
        continue
    frame_counter += 1
    frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_AREA)

    if saliency is None:
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()

    (success, saliencyMap) = saliency.computeSaliency(frame)
    saliencyMap = (saliencyMap * 255).astype('uint8')

    _, threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # threshMap = cv2.adaptiveThreshold(saliencyMap.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                   cv2.THRESH_BINARY,
    #                                   11, 2)

    kernel = np.ones((3, 3), 'uint8')
    threshMap = cv2.dilate(threshMap, kernel, iterations=1)
    threshMap = negative(threshMap)

    #

    kernel = np.ones((5, 5), 'uint8')
    threshMap = cv2.dilate(threshMap, kernel, iterations=1)

    contours, _ = cv2.findContours(threshMap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cont_rects = list(map(cv2.boundingRect, contours))
    for contour_idx in range(len(contours)):
        if cv2.contourArea(contours[contour_idx]) > 1000:
            cv2.drawContours(frame, contours, contour_idx, (255, 255, 0), 2)
            x, y, w, h = cont_rects[contour_idx]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
    # cv2.drawContours(frame, contours, -1, (255, 0, 0), 3)

    if frame_counter % 2 == light_frame:
        cv2.imshow('Frame', frame)
    cv2.imshow("Map", saliencyMap)
    cv2.imshow("threshMap", threshMap)
    # cv2.imshow("Real", frame_real)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

from imutils.video import VideoStream
import imutils
import cv2
import time
import numpy as np

# image = cv2.imread('/home/youngdanon/Документы/cv/test/test.jpg')
cap = cv2.VideoCapture('/dev/video0')
saliency = None


def negative(img):
    imagem = cv2.bitwise_not(img)
    return imagem


while True:
    ok, frame = cap.read()
    frame = cv2.GaussianBlur(frame, (1, 1), 0)
    if not ok:
        continue
    frame = imutils.resize(frame, width=500)
    if saliency is None:
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()

    (success, saliencyMap) = saliency.computeSaliency(frame)
    saliencyMap = (saliencyMap * 200).astype('uint8')

    threshMap = cv2.adaptiveThreshold(saliencyMap.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY,
                                      11, 2)

    kernel = np.ones((1, 1), 'uint8')
    threshMap = cv2.dilate(threshMap, kernel, iterations=1)
    threshMap = negative(threshMap)

    # _, threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), 'uint8')
    threshMap = cv2.dilate(threshMap, kernel, iterations=1)

    contours, _ = cv2.findContours(threshMap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cont_rects = list(map(cv2.boundingRect, contours))
    for contour_idx in range(len(contours)):
        if cv2.contourArea(contours[contour_idx]) > 1000:
            cv2.drawContours(frame, contours, contour_idx,  (255, 255, 0), 2)
            x, y, w, h = cont_rects[contour_idx]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
    # cv2.drawContours(frame, contours, -1, (255, 0, 0), 3)

    cv2.imshow("Frame", frame)
    cv2.imshow("Map", saliencyMap)
    cv2.imshow("threshMap", threshMap)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
cap.release()

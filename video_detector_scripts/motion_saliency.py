from imutils.video import VideoStream
import imutils
import time
import cv2

# initialize the motion saliency object and start the video stream
saliency = None
# cap = VideoStream(src='/dev/video4').start()
cap = cv2.VideoCapture('/dev/video2')
time.sleep(2.0)

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to 500px (to speedup processing)
    ok, frame = cap.read()
    if not ok:
        continue
    frame = imutils.resize(frame, width=500)

    # if our saliency object is None, we need to instantiate it
    if saliency is None:
        saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
        saliency.setImagesize(frame.shape[1], frame.shape[0])
        saliency.init()

    # convert the input frame to grayscale and compute the saliency
    # map based on the motion model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (success, saliencyMap) = saliency.computeSaliency(gray)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    threshMap = cv2.adaptiveThreshold(saliencyMap.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY,
                                      11, 2)

    # display the image to our screen
    cv2.imshow("Frame", frame)
    # cv2.imshow("Map", saliencyMap)
    cv2.imshow("threshMap", threshMap)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

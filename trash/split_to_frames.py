import cv2

path = 'output(3).avi'

cap = cv2.VideoCapture(path)
while True:
    ok, frame = cap.read()
    if not ok:
        print("Херь")
        continue
    cv2.imshow('output', frame)
    cv2.waitKey(0)

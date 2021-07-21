import cv2
import numpy

link = '/home/youngdanon/Документы/cv/test/bud_bottle.mp4'

cap = cv2.VideoCapture(link)

ret, frame = cap.read()
h, w, _ = frame.shape
writer = cv2.VideoWriter(
    'output.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),  # codec
    60,  # fps
    (w, h),  # width, height
    isColor=True)

while True:
    key = cv2.waitKey(1) & 0xFF
    ok, frame = cap.read()
    if not ok or key == ord("q"):
        print('govno')
        break

    cv2.imshow('wwer', frame)

    if frame.mean(axis=0).mean(axis=0)[0] < 10:
        continue
    writer.write(frame)

cap.release()
writer.release()

import cv2

link = ''

cap = cv2.VideoCapture(link)

ret, frame = cap.read()
h, w = frame.shape
writer = cv2.VideoWriter(
    '/video/output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),  # codec
    60,  # fps
    (w, h),  # width, height
    isColor=True)

while True:
    ret, frame = cap.read()
    if not ret and frame.mean(axis=0).mean(axis=0)[0] < 10:
        break
    writer.write(frame)
    
cap.release()
writer.release()

import cv2
cap = cv2.VideoCapture('/home/youngdanon/Загрузки/test_dataset_2021-05-25_1.mp4')
print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
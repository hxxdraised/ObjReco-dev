import cv2
from PIL import Image
import numpy as np
import tensorflow as tf


def read_tensor_from_readed_frame(frame, input_height=224, input_width=224,
                                  input_mean=0, input_std=255):
    output_name = "normalized"
    float_caster = tf.cast(frame, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def VideoSrcInit(paath):
    cap = cv2.VideoCapture(paath)
    flag, image = cap.read()
    if flag:
        print("Valid Video Path. Lets move to detection!")
    else:
        raise ValueError("Video Initialization Failed. Please make sure video path is valid.")
    return cap


def main():
    Labels_Path = "labels.txt"
    Model_Path = "/home/youngdanon/Загрузки/model.tflite"
    input_path = "canned_food.mp4"

    ##Loading labels
    labels = load_labels(Labels_Path)

    ##Load tflite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=Model_Path)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']

    ##Read video
    cap = VideoSrcInit(input_path)

    while True:
        ok, cv_image = cap.read()
        if not ok:
            break

        ##Converting the readed frame to RGB as opencv reads frame in BGR
        image = Image.fromarray(cv_image).convert('RGB')

        ##Converting image into tensor
        image_tensor = read_tensor_from_readed_frame(image, 224, 224)

        ##Test model
        interpreter.set_tensor(input_details[0]['index'], image_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        ## You need to check the output of the output_data variable and
        ## map it on the frame in order to draw the bounding boxes.

        cv2.namedWindow("cv_image", cv2.WINDOW_NORMAL)
        cv2.imshow("cv_image", cv_image)

        ##Use p to pause the video and use q to termiate the program
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            cv2.waitKey(0)
            continue
    cap.release()


if __name__ == '__main__':
    main()

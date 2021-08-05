import cv2
import numpy as np
from PIL import Image

CLASSES_PATH = "data/obj.names"
WEIGHTS_PATH = "weights/yolov4-tiny-detector_10000.weights"
CFG_PATH = "cfg/custom-yolov4-tiny-detector-test.cfg"


class YOLOTiny:
    def __init__(self, threshold=0.5, cfg_path=CFG_PATH, weights_path=WEIGHTS_PATH, classes_path=CLASSES_PATH):
        self.CLASSES = open(classes_path).read().strip().split('\n')
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.THRESHOLD = threshold
        self.COLORS = np.random.randint(0, 255, size=(len(self.CLASSES), 3), dtype=np.uint8)

        # determine the output layer
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, img):
        # construct a blob from the image
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        self.net.setInput(blob)
        outputs = self.net.forward(self.ln)

        boxes = []
        confidences = []
        class_ids = []
        h, w = img.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        results = []
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0]/w, boxes[i][1]/h)
                (width, height) = (boxes[i][2]/w, boxes[i][3]/h)
                box = [y, x, y+height, x+width]

                result = {
                    'bounding_box': box,
                    'class_id': class_ids[i],
                    'score': confidences[i]
                }
                results.append(result)
        return results

    def draw_results(self, image):
        results = self.detect_objects(image)

        # Plot the detection results on the input image
        original_image_np = image.astype(np.uint8)
        for obj in results:
            # Convert the object bounding box from relative coordinates to absolute
            # coordinates based on the original image resolution
            ymin, xmin, ymax, xmax = obj['bounding_box']
            xmin = int(xmin * original_image_np.shape[1])
            xmax = int(xmax * original_image_np.shape[1])
            ymin = int(ymin * original_image_np.shape[0])
            ymax = int(ymax * original_image_np.shape[0])

            # Find the class index of the current object
            class_id = int(obj['class_id'])

            # Draw the bounding box and label on the image
            color = [int(c) for c in self.COLORS[class_id]]
            cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
            # Make adjustments to make the label visible for all objects
            y = ymin - 15 if ymin - 15 > 15 else ymin + 15
            label = "{}: {:.0f}%".format(self.CLASSES[class_id], obj['score'] * 100)
            cv2.putText(original_image_np, label, (xmin, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show the final image
        original_uint8 = original_image_np.astype(np.uint8)
        Image.fromarray(original_uint8).show()


if __name__ == "__main__":
    yolo = YOLOTiny(threshold=0.5)
    img = cv2.imread("test_hard.png")
    print(yolo.detect_objects(img))
    yolo.draw_results(img)

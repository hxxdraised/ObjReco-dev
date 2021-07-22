import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

MODEL_PATH = "./weights/model.tflite"
CLASSES_PATH = "./cfg/tflite.class"


class TFLite:
    def __init__(self, model_path=MODEL_PATH, classes_path=CLASSES_PATH, threshold=0.5):
        self.THRESHOLD = threshold
        self.CLASSES = [classname for classname in open(CLASSES_PATH, "r").readlines()]
        self.COLORS = np.random.randint(0, 255, size=(len(self.CLASSES), 3), dtype=np.uint8)

        self._interpreter = tf.lite.Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()
        _, self._input_height, self._input_width, _ = self._interpreter.get_input_details()[0]['shape']

    def _preprocess_image(self, image):
        """Preprocess the input image to feed to the TFLite model"""
        img = tf.convert_to_tensor(image)
        # img = tf.io.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.uint8)

        resized_img = tf.image.resize(img, (self._input_height, self._input_width))
        resized_img = resized_img[tf.newaxis, :]
        return resized_img

    def _set_input_tensor(self, image):
        """Set the input tensor."""
        tensor_index = self._interpreter.get_input_details()[0]['index']
        input_tensor = self._interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def _get_output_tensor(self, index):
        """Return the output tensor at the given index."""
        output_details = self._interpreter.get_output_details()[index]
        tensor = np.squeeze(self._interpreter.get_tensor(output_details['index']))
        return tensor

    def detect_objects(self, image):
        """Returns a list of detection results, each a dictionary of object info."""
        preprocessed_image = self._preprocess_image(image)

        # Feed the input image to the model
        self._set_input_tensor(preprocessed_image)
        self._interpreter.invoke()

        # Get all outputs from the model
        boxes = self._get_output_tensor(0)
        classes = self._get_output_tensor(1)
        scores = self._get_output_tensor(2)
        count = int(self._get_output_tensor(3))

        results = []
        for i in range(count):
            if scores[i] >= self.THRESHOLD:
                result = {
                    'bounding_box': boxes[i],
                    'class_id': classes[i],
                    'score': scores[i]
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
    model = TFLite(threshold=0.3)
    img = cv2.imread("test_image.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model.draw_results(img)

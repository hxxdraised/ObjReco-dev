import os
import argparse
import cv2

from object_crop import get_object_from_frame


def process_frames(from_path, to_path, process_func, process_type):
    if not os.path.exists(to_path):
        os.mkdir(os.path.join(to_path))

    for root, dirs, filenames in os.walk(from_path):
        dirname = root.split("\\")[-1]
        print("Processing {}".format(dirname))
        for filename in filenames:
            img_from_path = os.path.join(root, filename).replace("\\", "/")
            img_to_path = os.path.join(to_path, dirname, filename).replace("\\", "/").replace("jpg", "png")

            processed_frame = process_func(img_from_path, process_type)

            if not os.path.exists(os.path.join(to_path, dirname).replace("\\", "/")):
                os.mkdir(os.path.join(to_path, dirname).replace("\\", "/"))
            cv2.imwrite(img_to_path, processed_frame)


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="path to file with frames")
    ap.add_argument("-o", "--output", required=True,
                    help="path to folder with cropped_objects")
    ap.add_argument("-p", "--process_type", required=True,
                    help="which way images are processed")
    args = vars(ap.parse_args())

    process_frames(args['input'], args["output"], get_object_from_frame, args["process_type"])

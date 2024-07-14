# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time
import threading

from alive.src.anti_spoof_predict import AntiSpoofPredict
from alive.src.generate_patches import CropImage
from alive.src.utility import parse_model_name

warnings.filterwarnings('ignore')

TARGET_WIDTH = 300
TARGET_HEIGHT = 225

def load_models(model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    models = {}
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        models[model_name] = {
            "h_input": h_input,
            "w_input": w_input,
            "scale": scale,
            "model_path": os.path.join(model_dir, model_name)
        }
    return model_test, models

def process_frame(frame, model_test, image_cropper, models):
    image_bbox = model_test.get_bbox(frame)
    prediction = np.zeros((1, 3))
    test_speed = 0

    def process_single_model(model_name, model_info):
        nonlocal prediction, test_speed
        param = {
            "org_img": frame,
            "bbox": image_bbox,
            "scale": model_info["scale"],
            "out_w": model_info["w_input"],
            "out_h": model_info["h_input"],
            "crop": True,
        }
        if model_info["scale"] is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, model_info["model_path"])
        test_speed += time.time() - start

    threads = []
    for model_name, model_info in models.items():
        thread = threading.Thread(target=process_single_model, args=(model_name, model_info))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    if label == 1:
        result_text = "Real: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        result_text = "Fake: {:.2f}".format(value)
        color = (0, 0, 255)

    cv2.rectangle(
        frame,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        frame,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5 * frame.shape[0] / 1024, color)

    return label

def test_rtmp_stream(rtmp_url, model_dir, device_id):
    model_test, models = load_models(model_dir, device_id)
    image_cropper = CropImage()

    cap = cv2.VideoCapture(rtmp_url)

    if not cap.isOpened():
        print("Error: Could not open RTMP stream.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to the target resolution
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

        processed_frame, result_text, label = process_frame(frame, model_test, image_cropper, models)

    return label
    #     # Display the resulting frame
    #     cv2.imshow('RTMP Stream', processed_frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--rtmp_url",
        type=str,
        default="rtmp://101.200.135.114:9090/live/live1",
        help="RTMP stream URL")
    args = parser.parse_args()
    test_rtmp_stream(args.rtmp_url, args.model_dir, args.device_id)

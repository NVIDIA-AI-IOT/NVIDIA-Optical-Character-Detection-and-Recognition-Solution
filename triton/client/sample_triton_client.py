import argparse
import sys
import json

import numpy as np
import tritonclient.http as httpclient
import cv2


def visualize(image, box, score, text):
    box_arr = np.array(box).reshape([4, 2]).astype(int)
    center = np.mean(box_arr, axis=0).astype(int)
    
    for i in range(4):
        x1 = box_arr[i][0]
        y1 = box_arr[i][1]
        x2 = box_arr[(i + 1) % 4][0]
        y2 = box_arr[(i + 1) % 4][1]

        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
    
    cv2.putText(image, f"{score:.2f}", box_arr[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(image, text, center, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 1)


if __name__ == "__main__":
    url = "localhost:8000"
    name = "nvocdr"

    triton_client = httpclient.InferenceServerClient(
            url=url, concurrency=1
    )

    if not triton_client.is_server_ready():
        print("server not ready!")
        exit(1)
    
    server_meta = triton_client.get_server_metadata()
    print(server_meta)

    model_meta = triton_client.get_model_metadata(name)
    print(model_meta)

    model_config = triton_client.get_model_config(name)
    print(model_config)

    img = cv2.imread("samples/test_img/super_resolution.jpg")
    print(img.shape)
    img_input = img.reshape([-1])
    print(img_input.shape)


    inputs = [httpclient.InferInput("INPUT_DATA", img_input.shape, "UINT8")]
    inputs[0].set_data_from_numpy(img_input)

    infer_res = triton_client.infer(name, inputs)

    boxes = infer_res.as_numpy("OUTPUT_BOX")
    scores = infer_res.as_numpy("OUTPUT_CONF")
    texts = infer_res.as_numpy("OUTPUT_TEXT")

    for idx, (box, score, txt) in enumerate(zip(boxes, scores, texts)):
        txt = txt.decode('utf-8')        
        visualize(img, box, score, txt)
    
    cv2.imwrite("test.png", img)

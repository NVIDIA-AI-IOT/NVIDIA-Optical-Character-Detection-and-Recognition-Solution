# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os, sys
import numpy as np
import json
import tritongrpcclient
import argparse
from nvjpeg import NvJpeg
import cv2


def readImageOpenCV(imgPath):
    img = cv2.imread(imgPath)
    return img


def getImagesPath(imgDir):
    allImgsPath = []
    supported_img_format = ['.jpg', '.jpeg', '.png']
    img_paths = os.listdir(imgDir)
    for img_path in img_paths:
        filename, img_ext = os.path.splitext(img_path)
        if img_ext in supported_img_format:
            allImgsPath.append(os.path.join(imgDir,img_path))   
    print(f'[nvOCDR] Find total {len(allImgsPath)} images in {imgDir}')
    return allImgsPath

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        required=False,
                        default="nvOCDR",
                        help="Model name")
    parser.add_argument('-d',
                        "--imagesDir",
                        type=str,
                        required=True,
                        help="Path to the images directory")
    parser.add_argument('-bs',
                        "--batchSize",
                        type=int,
                        required=True,
                        help="infer batch size")
    parser.add_argument("--url",
                        type=str,
                        required=False,
                        default="localhost:8001",
                        help="Inference server URL. Default is localhost:8001.")
    parser.add_argument('-v',
                        "--verbose",
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    args = parser.parse_args()

    try:
        triton_client = tritongrpcclient.InferenceServerClient(
            url=args.url, verbose=args.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    batchSize = args.batchSize
    inputs = []
    outputs = []
    input_data_name = "INPUT_DATA"
    input_img_extension = "INPUT_IMG_EXTENSION"
    output_img_name = "OUTPUT_IMG"
    output_predicts = "OUTPUT_TEXT_AND_BOX"
    inputs.append(tritongrpcclient.InferInput(input_data_name, (batchSize,1), "BYTES"))
    inputs.append(tritongrpcclient.InferInput(input_img_extension, (batchSize,1), "BYTES"))
    outputs.append(tritongrpcclient.InferRequestedOutput(output_img_name))
    outputs.append(tritongrpcclient.InferRequestedOutput(output_predicts))

    
    imgPaths = getImagesPath(args.imagesDir)
    if len(imgPaths) == 0:
        print(f'can not find images in : {args.imagesDir}')
        sys.exit(1)
    
    inferTimes, remainder = divmod(len(imgPaths), batchSize)
    if remainder != 0:
        print(f"[WARNING] The images number {len(imgPaths)} isn't evenly divisible by batchSize {batchSize}, will repreate the last image to satify the batch size.")
        imgPaths += [imgPaths[-1]] * ((inferTimes+1)*batchSize - len(imgPaths))

    resultPath = os.path.join(args.imagesDir, 'results')
    if not os.path.isdir(resultPath):
        os.makedirs(resultPath,exist_ok=True)

    imgData = []
    batchImgExt = []
    nj = NvJpeg()

    for i, imgPath in enumerate(imgPaths):
        imgExt = os.path.splitext(imgPath)[-1]
        batchImgExt.append(imgExt)
        if imgExt in ['.jpg', '.jpeg']:
            img = nj.read(imgPath)
            img_encode = nj.encode(img)
        else:
            img = cv2.imread(imgPath)
            img_encode = cv2.imencode(imgExt, img)[1].tobytes()
        
        print(f'[nvOCDR] Processing for: {imgPath}, image size: {img.shape}')

        imgData.append(img_encode)
        if len(imgData) == batchSize:
            inData = np.array(imgData).reshape(batchSize,-1)
            inImgExt = np.array(list(map(lambda x: x.encode("utf-8"), batchImgExt)),dtype=np.object_).reshape(batchSize,-1)
            inputs[0].set_data_from_numpy(inData)
            inputs[1].set_data_from_numpy(inImgExt)
           
            results = triton_client.infer(model_name=args.model_name,
                                        inputs=inputs,
                                        outputs=outputs)

            output0_data = results.as_numpy(output_img_name)
            output0_data = [nj.decode(output0_data[i][0]) for i in range(len(output0_data))]

            predict_text_box = results.as_numpy(output_predicts)
            predict_text_box = list(map(lambda x:x[0].decode("utf-8"), predict_text_box))
            predict_text_box_decode = [json.loads(predict) for predict in predict_text_box]
            for k, output in enumerate(output0_data):
                outputImgPath =  os.path.join(resultPath, os.path.basename( imgPaths[int(i/batchSize)*batchSize + k] + '_nvocdr_vis.jpg'))
                nj.write(outputImgPath, output)

                outputTextBoxPath = os.path.join(resultPath, os.path.basename( imgPaths[int(i/batchSize)*batchSize + k] + '_nvocdr_text_boxes.txt'))
                with open(outputTextBoxPath,'w', encoding='utf-8') as f:
                    for predicts in predict_text_box_decode[k]:
                        f.write(f"{predicts['text']}, {predicts['poly']}\n")

            batchImgExt = []
            imgData = []
    print(f'[nvOCDR] Inference done, results are save to :{resultPath}')

    # maxs = np.argmax(output0_data, axis=1)
    # print(maxs)
    # print("Result is class: {}".format(labels_dict[maxs[0]]))

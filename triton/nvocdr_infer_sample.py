

from mimetypes import init
import os
import sys
os.environ['PYTHONPATH'] = r'D:\03_Workspace\01_TAO\11_nvOCDR\NVIDIA-Optical-Character-Detection-and-Recognition-Solution\triton'
# change below path to yours
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\lib\x64')
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin')
os.add_dll_directory(r'D:\01_Software\opencv\opencv\build\x64\vc16\lib')
os.add_dll_directory(r'D:\01_Software\opencv\opencv\build\x64\vc16\bin')

import numpy as np

import json
import io
import argparse
from nvjpeg import NvJpeg
from utils.process import OCDRProcess
import cv2

from nvocdr_pybind import *

print('Load nvocdr module successfully')

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


class Model:
    def __init__(self, args):
        
        with open(args.spec,'r') as f:
            configs = json.load(f)

        param = nvOCDRParam()
        param.input_data_format = DataFormat.NHWC if configs['input_data_format'] == "NHWC" else DataFormat.NCHW
        self.ocdnet_trt_engine_path = configs['ocdnet_trt_engine_path']
        self.ocdnet_infer_input_shape = configs['ocdnet_infer_input_shape']
        param.ocdnet_binarize_threshold = configs['ocdnet_binarize_threshold']
        param.ocdnet_polygon_threshold = configs['ocdnet_polygon_threshold']
        param.ocdnet_unclip_ratio = configs['ocdnet_unclip_ratio']
        param.ocdnet_max_candidate = configs['ocdnet_max_candidate']
        param.upsidedown = configs['upsidedown']
        param.ocrnet_decode = OCRNetDecode.Attention if configs["ocrnet_decode"] == "Attention" else OCRNetDecode.CTC
        self.ocrnet_trt_engine_path = configs['ocrnet_trt_engine_path']
        self.ocrnet_dict_file = configs['ocrnet_dict_file']
        self.ocrnet_infer_input_shape = configs['ocrnet_infer_input_shape']
        self.nvOCDR_config = param
        self.nvOCDR = nvOCDRWarp(param, self.ocdnet_trt_engine_path, self.ocrnet_trt_engine_path, self.ocrnet_dict_file, self.ocdnet_infer_input_shape, self.ocrnet_infer_input_shape)
        if configs['input_data_format'] == "NHWC":
            self.input_h = configs['ocdnet_infer_input_shape'][1]
            self.input_w = configs['ocdnet_infer_input_shape'][2]
        else:
            self.input_h = configs['ocdnet_infer_input_shape'][2]
            self.input_w = configs['ocdnet_infer_input_shape'][3]

        
        self.is_high_resolution = configs['is_high_resolution_input']
        self.overlapRate = configs['overlapRate']
        assert self.overlapRate>0.1 and self.overlapRate<1.0, "[ERROR] overlapRate must be in (0.1,1), please modify this config in spec.json"
        self.ocdrProcess = OCDRProcess(self.input_w, self.input_h,  configs)
        self.nj = NvJpeg()

    def infer(self, inputDir, batchSize):
        print("nvOCDR infer...")
        
        imgPaths = getImagesPath(inputDir)
        if len(imgPaths) == 0:
            print(f'can not find images in : {inputDir}')
            sys.exit(1)
    
        inferTimes, remainder = divmod(len(imgPaths), batchSize)
        if remainder != 0:
            print(f"[WARNING] The images number {len(imgPaths)} isn't evenly divisible by batchSize {batchSize}, will repreate the last image to satify the batch size.")
            imgPaths += [imgPaths[-1]] * ((inferTimes+1)*batchSize - len(imgPaths))
           
        resultPath = os.path.join(inputDir, '../nvocdr_results')
        if not os.path.isdir(resultPath):
            os.makedirs(resultPath, exist_ok=True)
            
        oriImgs = []
        batchImgExt = []
        output_vis_images = []
        for i, imgPath in enumerate(imgPaths):
            imgExt = os.path.splitext(imgPath)[-1]
        
            if imgExt in ['.jpg', '.jpeg']:
                img = self.nj.read(imgPath)

            else:
                img = cv2.imread(imgPath)
            print(f'[nvOCDR] Processing for: {imgPath}, image size: {img.shape} ')
            oriImgs.append(img)
            if len(oriImgs) == batchSize:
                np_buffer, new_w, new_h, preprocessImgs = self.ocdrProcess.preprocess(oriImgs)
                
                if self.is_high_resolution:

                    # need to resize ori imgs according to keep ar scale, in order to match merged mask output size
                    results = self.nvOCDR.warpPatchInfer(preprocessImgs, np_buffer, self.overlapRate)
                else:

                    results = self.nvOCDR.warpInfer(np_buffer)
                print(f"[nvOCDR] Infer done, got {sum([len(r) for r in results])} texts ")
                output_vis_images, predict_text_box = self.ocdrProcess.postprocess(oriImgs, results, new_w ,new_h)

                
                predict_text_box_decode = [json.loads(predict) for predict in predict_text_box]
                for k, output in enumerate(output_vis_images):
                    outputImgPath =  os.path.join(resultPath, os.path.basename( imgPaths[int(i/batchSize)*batchSize + k] + '_nvocdr_vis.jpg'))

                    self.nj.write(outputImgPath, output)


                    outputTextBoxPath = os.path.join(resultPath, os.path.basename( imgPaths[int(i/batchSize)*batchSize + k] + '_nvocdr_text_boxes.txt'))
                    with open(outputTextBoxPath,'w', encoding='utf-8') as f:
                        for predicts in predict_text_box_decode[k]:
                            f.write(f"{predicts['text']}, {predicts['poly']}\n")
                oriImgs = []
                output_vis_images = []
        print(f'[nvOCDR] Inference done, results are save to :{resultPath}')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True, help="path to nvocdr spec file")
    parser.add_argument('--inputDir', required=True, help="path to input image folder")
    parser.add_argument('-b', '--batchSize', type=int, required=True, help="infer batchSize")
    args, unknown = parser.parse_known_args()
    model = Model(args)
    
    model.infer(args.inputDir, args.batchSize)
    
    
    

if __name__ == "__main__":
    main()


    
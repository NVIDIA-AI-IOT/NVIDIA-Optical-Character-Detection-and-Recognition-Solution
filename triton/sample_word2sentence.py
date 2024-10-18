import os, sys
import numpy as np
import json
import cv2
import argparse

from nvocdr import *
from utils.process import OCDRProcess

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

class NVOCDR:

    def __init__(self, model_cofig) -> None:
        with open(model_cofig,'r') as f:
            self.model_config = json.load(f)
        print(self.model_config )

        param = nvOCDRParam()
        param.input_data_format = DataFormat.NHWC if self.model_config['input_data_format'] == "NHWC" else DataFormat.NCHW
        self.ocdnet_trt_engine_path = self.model_config['ocdnet_trt_engine_path']
        self.ocdnet_infer_input_shape = self.model_config['ocdnet_infer_input_shape']
        param.ocdnet_binarize_threshold = self.model_config['ocdnet_binarize_threshold']
        param.ocdnet_polygon_threshold = self.model_config['ocdnet_polygon_threshold']
        param.ocdnet_unclip_ratio = self.model_config['ocdnet_unclip_ratio']
        param.ocdnet_max_candidate = self.model_config['ocdnet_max_candidate']
        param.upsidedown = self.model_config['upsidedown']
        param.ocrnet_decode = OCRNetDecode.Attention if self.model_config["ocrnet_decode"] == "Attention" else OCRNetDecode.CTC
        self.ocrnet_trt_engine_path = self.model_config['ocrnet_trt_engine_path']
        
        self.ocrnet_dict_file = self.model_config['ocrnet_dict_file']
        self.ocrnet_infer_input_shape = self.model_config['ocrnet_infer_input_shape']
        self.nvOCDR_config = param
        self.nvOCDR = nvOCDRWarp(param, self.ocdnet_trt_engine_path, self.ocrnet_trt_engine_path, self.ocrnet_dict_file,self.ocdnet_infer_input_shape, self.ocrnet_infer_input_shape)


        self.ocd_input_h = self.model_config['ocdnet_infer_input_shape'][1]
        self.ocd_input_w = self.model_config['ocdnet_infer_input_shape'][2]


        
        self.is_high_resolution = self.model_config['is_high_resolution_input']
        self.overlapRate = self.model_config['overlapRate']
        assert self.overlapRate>0.1 and self.overlapRate<1.0, "[ERROR] overlapRate must be in (0.1,1), please modify this config in spec.json"
        self.ocdrProcess = OCDRProcess(self.ocd_input_w, self.ocd_input_h,  self.model_config)
        self.word2sentence = Word2Sentence()

    def run(self, batch: list):
        
        np_buffer, new_w, new_h, preprocessImgs = self.ocdrProcess.preprocess(batch)
        if self.is_high_resolution:
            # need to resize ori imgs according to keep ar scale, in order to match merged mask output size
            results = self.nvOCDR.warpPatchInfer(preprocessImgs, np_buffer, self.overlapRate)
        else:
            results = self.nvOCDR.warpInfer(np_buffer)
        print(f"[nvOCDR] Infer done, got {sum([len(r) for r in results])} texts ")
        results_img, results_encode = self.ocdrProcess.postprocess(batch, results, new_w ,new_h)
        predict_text_box_decode = [json.loads(predict) for predict in results_encode]

        texts = []
        boxes = []
        papers = []
        for k, output in enumerate(predict_text_box_decode):
            for box_text in output:
                texts.append(box_text['text'])
                boxes.append(box_text['poly'])
            sentences = self.word2sentence.extractSentence(texts, boxes)
            papers.append(sentences)
        return results_img, predict_text_box_decode, papers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
   
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

    parser.add_argument('-c',
                        "--model-config",
                        type=str,
                        required=True,
                        help="path to config file of nvOCDR")

    args = parser.parse_args()
    
    batchSize = args.batchSize
    model_config = args.model_config
    assert os.path.isfile(model_config), f"No such a file {model_config}"


    imgPaths = getImagesPath(args.imagesDir)
    
    if len(imgPaths) == 0:
        print(f'[nvOCDR] Can not find images in : {args.imagesDir}')
        sys.exit(1)
    
    inferTimes, remainder = divmod(len(imgPaths), batchSize)
    if remainder != 0:
        print(f"[WARNING] The images number {len(imgPaths)} isn't evenly divisible by batchSize {batchSize}, will repreate the last image to satify the batch size.")
        imgPaths += [imgPaths[-1]] * ((inferTimes+1)*batchSize - len(imgPaths))

    resultPath = os.path.join(args.imagesDir, 'results')
    if not os.path.isdir(resultPath):
        os.makedirs(resultPath,exist_ok=True)
    

    # nj = NvJpeg()
    nvOCDR = NVOCDR(model_config)

    imgData = []
    for i, imgPath in enumerate(imgPaths):
        imgExt = os.path.splitext(imgPath)[-1]
        img = cv2.imread(imgPath)
        assert img.shape[0] > 736 and img.shape[1] > 1280 , "[nvOCDR] For image which's height < 736 and width < 1280, please set 'is_high_resolution_input' to false in spec.json"
        print(f'[nvOCDR] Processing for: {imgPath}, image size: {img.shape}')
        imgData.append(img)
        if len(imgData) == batchSize:
            results_images, text , papers = nvOCDR.run(imgData)
            
            for k, output in enumerate(results_images):
                outputImgPath =  os.path.join(resultPath, os.path.basename( imgPaths[int(i/batchSize)*batchSize + k] + '_nvocdr_vis.jpg'))
                outputpaperPath =  os.path.join(resultPath, os.path.basename( imgPaths[int(i/batchSize)*batchSize + k] + '_sentence.txt'))
                
                cv2.imwrite(outputImgPath, output)
                print(f'[nvOCDR] Inference done, results are save to :{outputImgPath}')
                with open(outputpaperPath, 'w')  as f:
                    for sentence in papers[k]:
                        f.write(f'{sentence}\n')
                
                    
        imgData = []
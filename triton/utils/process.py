import os
import cv2
import json
import math
import numpy as np

from utils.cuda_resize_keep_AR import image_resize


def directResize(imgs, width, hight):
    resizeImgs = []
    for img in imgs:
        imgResize = cv2.resize(img,(width, hight))
        resizeImgs.append(imgResize)
    return np.asarray(resizeImgs)


def keepAspectRatioResize(imgs, width, hight):
    resizeImgs = []

    for img in imgs:
        imgResize = image_resize(img,width,hight)
        imgResize = imgResize.reshape((hight, width, 3)).astype(np.uint8)
        resizeImgs.append(imgResize)

    return np.asarray(resizeImgs)

def largeResolutionResize(imgs, width, hight):
    pass


class OCDRProcess:
    "preprocess and postprocess for nvOCDR triton sample"
    def __init__(self, inWidth, inHight, configs) -> None:

        self.inWidth = inWidth
        self.inHight = inHight

        self.is_high_resolution = configs['is_high_resolution_input']
        self.overlapRate = configs['overlapRate']
        if self.is_high_resolution:
            self.patch_w = self.inWidth
            self.patch_h = self.inHight
            self.overlap_w = int(self.overlapRate * self.patch_w)
            self.overlap_h = int(self.overlapRate * self.patch_h)
            self.keep_ar = True
        else:
            self.keep_ar = configs['resize_keep_aspect_ratio']
        self.font_size = configs['font_size']
        self.font_color = configs['font_color']

    def preprocess(self, imgs):
        new_w = 0
        new_h = 0
        raw_imgs = imgs
        if self.is_high_resolution:
            assert len(raw_imgs) == 1, 'patch method only support batch size = 1 '
            # crop ori image to a bunch of square patches
            raw_imgs = np.asarray(raw_imgs)
            ori_h = raw_imgs.shape[1]
            ori_w = raw_imgs.shape[2]
            ori_c = raw_imgs.shape[3]

            croppable_ori_w = int(math.ceil((ori_w - self.patch_w)/(self.patch_w - self.overlap_w)) *  (self.patch_w - self.overlap_w) + self.patch_w)
            croppable_ori_h = int(math.ceil((ori_h - self.patch_h)/(self.patch_h -self.overlap_h)) *  (self.patch_h -self.overlap_h) + self.patch_h)
            croppable_imgs = image_resize(raw_imgs, croppable_ori_w, croppable_ori_h, True)
            croppable_img = croppable_imgs[0]
            num_col_cut = int((croppable_ori_w- self.patch_w)/(self.patch_w - self.overlap_w))
            num_row_cut = int((croppable_ori_h- self.patch_h)/(self.patch_h -self.overlap_h))

            patchs = []
            for i in range(0, num_row_cut+1):
                for j in range(0, num_col_cut+1):
                    x_start = int(j*(self.patch_w - self.overlap_w))
                    y_start = int(i*(self.patch_h - self.overlap_h))
                    patch = croppable_img[y_start : y_start + self.patch_h, x_start : x_start + self.patch_w, :]
                    patchs.append(patch)

            resizeImgs = np.asarray(patchs)
            new_w = croppable_ori_w
            new_h = croppable_ori_h
            imgs = croppable_imgs
        else:
            resizeImgs = []
            for img in raw_imgs:
                img = np.expand_dims(img, axis=0)
                resizeImg = image_resize(img, self.inWidth, self.inHight, self.keep_ar)
                resizeImgs.append(resizeImg)
            resizeImgs = np.concatenate(resizeImgs,axis=0)
            new_w = self.inWidth
            new_h = self.inHight

        return resizeImgs, new_w, new_h, imgs


    def postprocess(self, imgs, predicts, new_w, new_h):
        visResults = []
        predicts_encode = []
        for i, predict in enumerate(predicts):
            vis, predict_encode = self.visAndEncode(imgs[i], predict, new_w, new_h)
            visResults.append(vis)
            predicts_encode.append(json.dumps(predict_encode))
        return visResults, predicts_encode


    def visAndEncode(self, image, predict, input_width, input_height):

        if self.keep_ar:
            im_w = float(image.shape[1])
            im_h = float(image.shape[0])
            aspect_ratio_input_image = im_w/im_h
            aspect_ratio_output_image = float(input_width)/float(input_height);
            
            new_height=0
            new_width=0
        
            if (aspect_ratio_input_image >= aspect_ratio_output_image):
                new_width = input_width
                new_height =  new_width / aspect_ratio_input_image
                scale_x = scale_y = im_w / new_width
            else:
                new_height = input_height
                new_width = aspect_ratio_input_image*new_height
                scale_x = scale_y =  im_h / new_height
        else:  
            scale_x = image.shape[1] / input_width
            scale_y = image.shape[0] / input_height
        pred_canvas = image.copy().astype(np.uint8)
        predict_encode = []
        for polygen, text in predict:
            poly = [polygen.x1, polygen.y1, polygen.x2, polygen.y2, polygen.x3, polygen.y3, polygen.x4, polygen.y4]
            box = np.array(poly).reshape(-1, 2).astype(np.float32)
            box[:,0] *=  scale_x
            box[:,1] *=  scale_y
            box = box.astype(np.int32)
            cv2.polylines(pred_canvas, [box], True, (0, 255, 0), 2)
           
            cv2.putText(pred_canvas, f'{text[0]}', tuple(np.min(box,axis=0)), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_color , 2)
            predict_encode.append({'text':text[0], 'text_conf':text[1],'poly':box.reshape(-1).tolist()})
        return pred_canvas, predict_encode

    def patchImg(self, oriImg):
        ori_h = oriImg.shape[1]
        ori_w = oriImg.shape[2]
        ori_c = oriImg.shape[3]

        # need to resize the ori image so that the there is no need to padding when crop the patch
        croppable_ori_w = int(math.ceil((ori_w - self.patch_w)/self.overlap_w) *  self.overlap_w + self.patch_w)
        croppable_ori_h = int(math.ceil((ori_h - self.patch_h)/self.overlap_h) *  self.overlap_h + self.patch_h)
        croppable_img = image_resize(oriImg, croppable_ori_w, croppable_ori_h, True)
        croppable_img = croppable_img[0]
        num_col_cut = int((croppable_ori_w- self.patch_w)/self.overlap_w)
        num_raw_cut = int((croppable_ori_h- self.patch_h)/self.overlap_h)
        patchs = []
        for i in range(0, num_raw_cut+1):
            for j in range(0, num_col_cut+1):
                x_start = int(j*self.overlap_w)
                y_start = int(i*self.overlap_h)
                patch = croppable_img[y_start : y_start + self.patch_h, x_start : x_start + self.patch_w, :]
                patchs.append(patch)
        patchs = np.asarray(patchs)
        return patchs, croppable_ori_w, croppable_ori_h, croppable_img



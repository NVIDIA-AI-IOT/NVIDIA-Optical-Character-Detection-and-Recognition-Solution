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

import numpy as np
import sys
import json
import io
import os

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils

from nvocdr import *
from nvjpeg import NvJpeg
from utils.process import OCDRProcess
import cv2


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output_img_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT_IMG")
        output_text_box_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT_TEXT_AND_BOX")

        # Convert Triton types to numpy types
        self.output_img_dtype = pb_utils.triton_string_to_numpy(
            output_img_config['data_type'])
        self.output_text_box_dtype = pb_utils.triton_string_to_numpy(
            output_text_box_config['data_type'])

        # read config file
        with open(os.path.dirname(__file__) + '/../spec.json','r') as f:
            configs = json.load(f)

        # set up the nvOCDR config
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
        self.nvOCDR = nvOCDRWarp(param, self.ocdnet_trt_engine_path, self.ocrnet_trt_engine_path, self.ocrnet_dict_file,self.ocdnet_infer_input_shape, self.ocrnet_infer_input_shape)
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

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            print("[nvOCDR] Infer...")
            oriImgs = []
            in_img = pb_utils.get_input_tensor_by_name(request, "INPUT_DATA")
            in_img_ext = pb_utils.get_input_tensor_by_name(request, "INPUT_IMG_EXTENSION")
            np_buffer = in_img.as_numpy()
            np_ext_buffer = in_img_ext.as_numpy()
            img_extensions = list(map(lambda x:x[0].decode("utf-8"), np_ext_buffer))

            for i in range(np_buffer.shape[0]):
                if img_extensions[i] in ['.jpg_gpu', '.jpeg_gpu']:
                    img_decode = self.nj.decode(np_buffer[i][0])
                else:
                    buff = np.fromstring(np_buffer[i][0], np.uint8)
                    buff = buff.reshape(1, -1)
                    img_decode = cv2.imdecode(buff, cv2.IMREAD_COLOR)
                oriImgs.append(img_decode)

            np_buffer, new_w, new_h, preprocessImgs = self.ocdrProcess.preprocess(oriImgs)
            
            if self.is_high_resolution:
                # need to resize ori imgs according to keep ar scale, in order to match merged mask output size
                results = self.nvOCDR.warpPatchInfer(preprocessImgs, np_buffer, self.overlapRate)
            else:
                results = self.nvOCDR.warpInfer(np_buffer)
            print(f"[nvOCDR] Infer done, got {sum([len(r) for r in results])} texts ")

            results_img, results_encode = self.ocdrProcess.postprocess(oriImgs, results, new_w ,new_h)
            output = []
            for img_idx, res in enumerate(results_img):
                if img_extensions[img_idx] in ['.jpg_gpu', '.jpeg_gpu']:
                    res_img_encode = self.nj.encode(res)
                else:
                    res_img_encode = cv2.imencode(img_extensions[img_idx], res)[1].tobytes()
                output.append(res_img_encode)
            text = np.array(output).reshape(len(results),-1)

            results_encode = np.array(list(map(lambda x: x.encode("utf-8") , results_encode)),dtype=np.object_).reshape(len(results),-1)
            
            out_img = pb_utils.Tensor("OUTPUT_IMG",
                                           text.astype(self.output_img_dtype))
            out_text_box = pb_utils.Tensor("OUTPUT_TEXT_AND_BOX",
                                           results_encode.astype(self.output_text_box_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_img, out_text_box])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')


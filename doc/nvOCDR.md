# nvOCDR C API Reference

## nvOCDR macro
### `MAX_CHARACTER_LEN`
The maximum length of the character to be recognized in nvOCDR library. Default to be `64`.

### `MAX_PLY_CNT`
The maximum number of points of the deteced text area polygons. Default to be `8`.

### `MAX_BATCH_SIZE`
The maximum number of images can be feed to nvOCDR at once. Default to be `64`
## nvOCDR enum
### `nvOCDRStat`
The status of execution of nvOCDR library.
|Value|Description|
|:---|:---|
|`SUCCESS`|Success in execution|
|`FAIL`|Fail in execution|

### `Device`
The data location.
|Value|Description|
|:---|:---|
|`CPU`|The data is stored on CPU|
|`GPU`|The data is stored on GPU|

**Notes**: Only supports data on GPU now.
### `DataFormat`
The input data format.
|Value|Description|
|:---|:---|
|`NCHW`|The data is stored in shape [batch, channel, height, width]|
|`NHWC`|The data is stored in shape [batch, height, width, channel]|

**Notes**: Only supports NHWC input now.
## nvOCDR meta struct

### `nvOCDRParam`
The parameters to initialize nvOCDR library.
|Variable|Type|Description|
|:---|:---|:---|
|`input_data_format`|`DataFormat`|The input data format to be processed in the nvOCDR library|
|`ocdnet_trt_engine_path`|`char*`|The absolute path to the OCDNet TensorRT engine|
|`ocdnet_binarize_threshold`|`float`|The threshold value to binarize the OCDNet's output mask|
|`ocdnet_polygon_threshold`|`float`|The threshold value to filter the polygons based on the confidence score. The confidence score of a polygon is the average value of every pixel's probability in the polygon mask|
|`ocdnet_max_candidate`|`int`|The maximum number of text area to be produced in one single image from OCDNet|
|`ocdnet_infer_input_shape`|`int32_t[3]`| The TensorRT engine inference shape of OCDNet. In `CHW` order |
|`upsidedown`|`bool`|Enable the nvOCDR to recognize the character that is totally upsidedown|
|`ocrnet_trt_engine_path`|`char*`|The absolute path to the OCRNet TensorRT engine|
|`ocrnet_dict_file`|`char*`| The absolute path to the OCRNet character list file|
|`ocrnet_infer_input_shape`|`int32_t[3]`| The TensorRT engine inference shape of OCRNet. In `CHW` order |

### `nvOCDRInput`
The input data structure of nvOCDR. All the data feed into the library should be wrapped in this struct.

|Variable|Type|Description|
|:---|:---|:---|
|`shape`|`int[4]`|The shape of input data. In `NHWC` format|
|`mem_ptr`|`void*`|The pointer of the input data.|
|`device_type`|`Device`|`GPU`|

### `nvOCDROutputBlob`
The single text output struct of nvOCDR, which contains the coordinates of the polygon's vertices that covers a text area and the corresponding decoded text.

|Variable|Type|Description|
|:---|:---|:---|
|`poly_cnt`|`int32_t`|The valid length of the array stores the coordinates|
|`polys`|`int32_t[MAX_PLY_CNT]`|The array stores|
|`mem_ptr`|`void*`|The pointer of the input data.|
|`device_type`|`Device`|`GPU`|

### `nvOCDROutputMeta`
The wrapped texts output struct for one inference of nvOCDR, which contains the `nvOCDROutputBlob`

|Variable|Type|Description|
|:---|:---|:---|
|`batch_size`|`int32_t`|The batch size of this inference|
|`text_cnt`|`int32_t[MAX_PLY_CNT]`|The array stores|
|`text_ptr`|`void*`|The pointer of the input data|

## nvOCDR function

### `nvOCDRp nvOCDR_init(nvOCDRParam param)`
The function to initialize the nvOCDR library. It will return a handle of nvOCDR library.

|Parameter|Type|Description|
|:---|:---|:---|
|`param`|`nvOCDRParam`|The parameters to initialize nvOCDR library|


### `nvOCDRStat nvOCDR_inference(nvOCDRInput input, nvOCDROutputMeta* output, nvOCDRp nvocdr_ptr)`
The function to do OCR. It takes batch images data on GPU as input and store the polygons and texts in the `output` of those images.

|Parameter|Type|Description|
|:---|:---|:---|
|`input`|`nvOCDRInput`|The parameters to initialize nvOCDR library|
|`output`|`nvOCDROutputMeta*`|The pointer to the output struct|
|`nvocdr_ptr`|`nvOCDRp`| The handle of nvOCDR library|

### `void nvOCDR_deinit(nvOCDRp nvocdr_ptr)`
The function to destroy the nvOCDR library resources when you don't need the library.

|Parameter|Type|Description|
|:---|:---|:---|
|`nvocdr_ptr`|`nvOCDRp`|The handle of nvOCDR library|

## Performance tuning guide

Besides the hardware the software running on, the performance(images/sec) of nvOCDR library is influenced a lot by the initial parameters. Here are two tips for better performance 

### Run with optimal batch size and shape.
When create TensorRT engine, there is a option called `optShape`. The generated engine will have the best performance with this shape. Thus, one should chose the optimal batch size and spatial shape according to the use cases:
- For OCDNet, the optimal batch size should be the number of input streams. For example, if there is only one single video stream to be processed in DeepStream pipeline, then batch_size=1 is the best choice.

- For OCRNet, one could chose the average number of text areas in one single image muliplied with the batch size of OCDNet as the batch size of OCRNet.

- For general performance concern, the batch size and spatial shape should be muliple of 16.

### Set OCDNet post-process parameters.

OCDNet postprocessing takes 20% percent of time. We could change the OCDNet postprocessing for better performance in case it won't bother the accuracy a lot.

- increase the `ocdnet_polygon_threshold` to filter out more polygons with lower confidence score.
- decrease the `ocdnet_max_candidates` to limit the number of OCDNet output to be post-processed.


### Disable `upsidedown`.

If in the use case, there is not much upsidedown characters, you could disable the `upsidedown` option. This option will bring in extra 30% compute workload. 


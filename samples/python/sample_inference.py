from nvocdr_python import *
import cv2
import argparse

def get_args():
    parser = argparse.ArgumentParser("nvocdr python binding test")
    parser.add_argument("--ocd_model", type=str, default="/home/csh/nvocdr/onnx_models/dcn_resnet18.engine", required=False)
    parser.add_argument("--ocr_model", type=str, default="/home/csh/nvocdr/onnx_models/ocrnet_resnet50.engine", required=False)
    parser.add_argument("--image", type=str, default="/home/csh/nvocdr/samples/test_img/scene_text.jpg", required=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    ocd_model = args.ocd_model
    ocr_model = args.ocr_model
    test_img = args.image
    test_img = cv2.imread(test_img, cv2.IMREAD_COLOR)
    ocd_param = nvOCDParam()
    ocd_param.type = OCD_MODEL_TYPE.OCD_MODEL_TYPE_NORMAL

    ocr_param = nvOCRParam()
    ocr_param.type = OCR_MODEL_TYPE.OCR_MODEL_TYPE_CTC

    process_param = ProcessParam()
    input_shape = test_img.shape
    print(input_shape)


    nvocdr = nvOCDRWarp( ocd_param, ocr_param, process_param, ocd_model, ocr_model, input_shape)




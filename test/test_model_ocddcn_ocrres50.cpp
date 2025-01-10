#include <gtest/gtest.h>

#include "nvocdr.h"
#include "test_utils.hpp"

/* test on OCD dcn + OCR reset50*/
TEST(TEST_OCD_DCN_OCR_RES50, NORMAL_IMAGE) {
    std::string img_dir = "/home/csh/nvocdr/samples/test_img/";
    std::string model_dir = "/home/csh/nvocdr/onnx_models";
    nvocdr::test::TestAssistant assitant(img_dir, model_dir);


    assitant.setup(nvocdr::test::TEST_IMG_CASE_SCENE_TEXT, nvocdr::test::TEST_MODEL_COMBO_DCN_RES50);
    void* nvocdr_handler = nvOCDR_initialize(assitant.getParam());

    nvOCDROutput output;
    auto input = assitant.getInput();
    nvOCDR_process(nvocdr_handler, input, &output);
    const auto gts = assitant.getGt();
    auto f1 = nvocdr::test::Metric::computeMetricF1(gts, output);
    ASSERT_GT(f1, 0.9);
    
}

TEST(TEST_OCD_DCN_OCR_RES50, SUPER_RESOLUTION_IMAGE) {
    // std::string img_dir = "/home/csh/nvocdr/samples/test_img/";
    // std::string model_dir = "/home/csh/nvocdr/onnx_models";
    // nvocdr::test::TestAssistant assitant(img_dir, model_dir);


    // assitant.setup(nvocdr::test::TEST_IMG_CASE_SCENE_TEXT, nvocdr::test::TEST_MODEL_COMBO_DCN_RES50);
    // void* nvocdr_handler = nvOCDR_initialize(assitant.getParam());

    // nvOCDROutput output;
    // auto input = assitant.getInput();
    // nvOCDR_process(nvocdr_handler, input, &output);
    // const auto gts = assitant.getGt();
    // auto f1 = nvocdr::test::Metric::computeMetricF1(gts, output);
    // ASSERT_GT(f1, 0.9);
}

TEST(TEST_OCD_DCN_OCR_RES50, SLIM_IMAGE) {
    
}

// TEST(TEST_OCD_)
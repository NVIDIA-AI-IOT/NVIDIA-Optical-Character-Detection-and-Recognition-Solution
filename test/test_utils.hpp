#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "nvocdr.h"

namespace nvocdr {
namespace test {

std::vector<std::string> split(std::string s, std::string delimiter) {
  size_t pos_start = 0, pos_end, delim_len = delimiter.length();
  std::string token;
  std::vector<std::string> res;

  while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
    token = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back(token);
  }

  res.push_back(s.substr(pos_start));
  return res;
}

std::vector<Text> load_gt(const std::string& gt_path) {
  std::vector<Text> ret;
  std::ifstream input(gt_path.c_str());
  for (std::string line; getline(input, line);) {
    ret.emplace_back();
    auto& gt_text = ret.back();
    auto comma_idx = line.find(',');
    std::string text = line.substr(0, comma_idx);
    if (text.length() > MAX_CHARACTER_LEN)
      continue;
    strcpy(gt_text.text, text.c_str());

    auto coords = split(line.substr(comma_idx + 1), ":");
    if (coords.size() != 8)
      continue;
    for (size_t i = 0; i < 8; ++i) {
      gt_text.polygon[i] = std::stof(coords[i]);
    }
  }
  std::cout << "get gt size: " << ret.size() << "\n";
  return ret;
}

cv::RotatedRect getRect(const Text& text) {
    // for(size_t i = 0; i< 6; ++i) {
    //     std::cout<< text.polygon[i] << " ";
    // } 
    // std::cout<< "\n";
  return cv::RotatedRect(cv::Point2f(text.polygon[0], text.polygon[1]),
                         cv::Point2f(text.polygon[2], text.polygon[3]),
                         cv::Point2f(text.polygon[4], text.polygon[5]));
}

enum TEST_IMG_CASE { TEST_IMG_CASE_SCENE_TEXT };
enum TEST_MODEL_COMBO { TEST_MODEL_COMBO_DCN_RES50 };

class TestAssistant {
 public:
  TestAssistant() = default;
  TestAssistant(const std::string& img_dir, const std::string& model_dir)
      : mImageDir(img_dir), mModelDir(model_dir) {};
  void setup(TEST_IMG_CASE img_case, TEST_MODEL_COMBO model_combo) {
    if (img_case == TEST_IMG_CASE_SCENE_TEXT) {
      mTestImage = cv::imread(std::filesystem::path(mImageDir) / "scene_text.jpg");
      mGtPath = std::filesystem::path(mImageDir) / "scene_text.txt";
    } else {
    }
    if (model_combo == TEST_MODEL_COMBO_DCN_RES50) {
      auto ocd_path = std::filesystem::path(mModelDir) / "dcn_resnet18.engine";
      auto ocr_path = std::filesystem::path(mModelDir) / "ocrnet_resnet50.engine";

      strcpy(mParam.ocd_param.model_file, ocd_path.c_str());
      strcpy(mParam.ocr_param.model_file, ocr_path.c_str());
    }
  }
  nvOCDRParam getParam() { return mParam; }
  nvOCDRInput getInput() {
    return {.height = static_cast<size_t>(mTestImage.rows),
            .width = static_cast<size_t>(mTestImage.cols),
            .num_channel = static_cast<size_t>(mTestImage.channels()),
            .data = mTestImage.data,
            .data_format = DATAFORMAT_TYPE_HWC};
  }

  std::vector<Text> getGt() {
    return load_gt(mGtPath);
  }

 private:
  std::string mImageDir;
  std::string mModelDir;
  cv::Mat mTestImage;
  std::string mGtPath;
  std::string mOCDModel;
  std::string mOCRModel;
  nvOCDRParam mParam;
};

class Metric {
 public:
  static float computeMetricF1(const std::vector<Text>& gts, const nvOCDROutput& preds,
                               float iou_threshold = 0.5) {
    // std::vector<float> iou(gts.size() * preds.num_texts, 0);

    // each gt's best matched pred
    std::vector<int> best_match_gt(gts.size(), 0);
    // each pred's best matched gt
    std::vector<int> best_match_pred(preds.num_texts, 0);
    std::vector<float> best_match_pred_iou(preds.num_texts, 0);

    for (size_t i = 0; i < gts.size(); ++i) {
      float cur_gt_best_iou = 0;
      int cur_gt_best_pred_idx = -1;
      const auto gt = getRect(gts[i]);

      for (size_t j = 0; j < preds.num_texts; ++j) {
        const auto pred = getRect(preds.texts[j]);

        std::vector<cv::Point> intersect;
        cv::rotatedRectangleIntersection(gt, pred, intersect);

        if(intersect.size() < 3){
            continue;
        } 

        float intersect_area = cv::contourArea(intersect);
        float union_area = gt.size.width * gt.size.height + pred.size.width * pred.size.height;

        float iou = intersect_area / (union_area - intersect_area);

        if (iou < iou_threshold) {
          continue;
        }
        if (iou > cur_gt_best_iou) {
          cur_gt_best_iou = iou;
          cur_gt_best_pred_idx = j;
        }

        if (iou > best_match_pred_iou[j]) {
          best_match_pred_iou[j] = iou;
          best_match_pred[j] = i;
        }
      }
      best_match_gt[i] = cur_gt_best_pred_idx;
    }

    float tp = 0;
    float fp = 0;

    for (size_t i = 0; i < preds.num_texts; ++i) {
      if (best_match_pred[i] >= 0 && best_match_gt[best_match_pred[i]] == i) {
        tp += 1;
      } else {
        fp += 1;
      }
    }

    float fn = 0;
    for(size_t i = 0; i < gts.size(); ++i) {
      if(best_match_gt[i] == -1) {
        fn += 1;
      }
    }
    float f1 = 2 * tp / (2 * tp + fp + fn);
    LOG(INFO) << "f1 " << f1;
    return f1;
  }
};

}  // namespace test

}  // namespace nvocdr

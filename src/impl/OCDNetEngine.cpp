#include <iostream>
#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "OCDNetEngine.h"
#include "timer.hpp"

namespace nvocdr {
inline void correctQuad(QUADANGLE& quadangle) {
  // https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/
  bool direction = cv::norm(quadangle[0] - quadangle[1]) <= cv::norm(quadangle[1] - quadangle[2]);
  static cv::Point2f tmp;
  if (!direction) {
    tmp = quadangle[3];
    quadangle[3] = quadangle[2];
    quadangle[2] = quadangle[1];
    quadangle[1] = quadangle[0];
    quadangle[0] = tmp;
  }
}

bool OCDNetEngine::customInit() {
  setupInput(OCDNET_INPUT, {}, true);

  // todo, unify output name, and remove this hack
  if (mParam.type == nvOCDParam::OCD_MODEL_TYPE::OCD_MODEL_TYPE_MIXNET) {
    mOutputName = "fy_preds";
  } else if (mParam.type == nvOCDParam::OCD_MODEL_TYPE::OCD_MODEL_TYPE_NORMAL) {
    mOutputName = "pred";
  }
  setupOutput(mOutputName, {}, true);

  return true;
}

float* OCDNetEngine::getMaskOutputBuf() {
  return reinterpret_cast<float*>(
      mBufManager.getBuffer(getBufName(mOutputName), BUFFER_TYPE::HOST));
}

size_t OCDNetEngine::getOutputChannelIdx() {
  if (mParam.type == nvOCDParam::OCD_MODEL_TYPE::OCD_MODEL_TYPE_MIXNET) {
    return 1;
  } else if (mParam.type == nvOCDParam::OCD_MODEL_TYPE::OCD_MODEL_TYPE_NORMAL) {
    return 0;
  }
  return 0;
}
size_t OCDNetEngine::getOutputChannels() {
  if (mParam.type == nvOCDParam::OCD_MODEL_TYPE::OCD_MODEL_TYPE_MIXNET) {
    return 4;
  } else if (mParam.type == nvOCDParam::OCD_MODEL_TYPE::OCD_MODEL_TYPE_NORMAL) {
    return 1;
  }
  return 0;
}
void OCDNetEngine::computeTextCandidates(const cv::Mat& mask, std::vector<QUADANGLE>* const quads,
                                         std::vector<Text>* const texts, size_t* num_text,
                                         const ProcessParam& process_param) {
  mTextCntrCandidates.clear();
  if (mParam.type == nvOCDParam::OCD_MODEL_TYPE::OCD_MODEL_TYPE_MIXNET) {
    computeTextCandidatesMixNet(mask, quads, texts, num_text, process_param);
  } else if (mParam.type == nvOCDParam::OCD_MODEL_TYPE::OCD_MODEL_TYPE_NORMAL) {
    computeTextCandidatesNormal(mask, quads, texts, num_text, process_param);
  }
}
void OCDNetEngine::computeTextCandidatesNormal(const cv::Mat& mask,
                                               std::vector<QUADANGLE>* const quads,
                                               std::vector<Text>* const texts, size_t* num_text,
                                               const ProcessParam& process_param) {
  std::vector<std::vector<cv::Point>> raw_contours;

  cv::findContours(mask, raw_contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

  // filter out too small
  std::copy_if(raw_contours.begin(), raw_contours.end(), std::back_inserter(mTextCntrCandidates),
               [&process_param](auto const& contour) {
                 return cv::contourArea(contour) >= process_param.min_pixel_area;
               });

  // sort by score, desc
  std::sort(mTextCntrCandidates.begin(), mTextCntrCandidates.end(),
            [](auto const& contour_a, auto const& contour_b) {
              float score_a = cv::contourArea(contour_a) / cv::boundingRect(contour_a).area();
              float score_b = cv::contourArea(contour_b) / cv::boundingRect(contour_b).area();
              return score_a > score_b;
            });

  for (size_t i = 0; i < std::min(mTextCntrCandidates.size(), process_param.max_candidate); ++i) {
    auto const& contour = mTextCntrCandidates[i];
    float polygon_score = cv::contourArea(contour) / cv::boundingRect(contour).area();
    if (polygon_score <= process_param.polygon_threshold) {
      break;
    }
    // !!! (todo) find better way to find minimum quadrilateral, use minAreaRect temporarily
    auto rotated = cv::minAreaRect(contour);

    // (todo) let user to finetune?
    static float long_side_scale = 2;
    static float short_side_scale = 1.1;

    cv::Size new_size = rotated.size;

    // enlarge the text area
    if (new_size.width > 2 * new_size.height) {
      new_size.height *= long_side_scale;
      new_size.width *= short_side_scale;
    } else if (new_size.height > 2 * new_size.width) {
      new_size.width *= long_side_scale;
      new_size.height *= short_side_scale;
    } else {
      new_size.width *= 2;
      new_size.height *= 2;
    }

    rotated = cv::RotatedRect(rotated.center, new_size, rotated.angle);

    // prepare output
    auto& vertices = (*quads)[*num_text];
    Text& text = (*texts)[*num_text];

    rotated.points(&vertices[0]);
    correctQuad(vertices);
#pragma unroll
    for (size_t j = 0; j < QUAD; ++j) {
      text.polygon[j * 2] = vertices[j].x;
      text.polygon[j * 2 + 1] = vertices[j].y;
    }
    *num_text += 1;
  }
}



// designed for mixnet, given the inner and outer text, and the inner fited line
inline bool findMatryoshkaText(const std::vector<cv::Point>& inner,
                               const std::vector<cv::Point>& outer, 
                               cv::Mat& viz, cv::Scalar color, cv::RotatedRect* rotated_rect, float min_pixel_area) {
  float ctr_area = cv::contourArea(inner);                          
  if (ctr_area <= min_pixel_area) {
    return false;
  }
  auto rrect = cv::minAreaRect(inner);
  float rrect_area = rrect.size.width * rrect.size.height;

  if (ctr_area / rrect_area < 0.5) {
    return false;
  }

  if (rrect.size.width < rrect.size.height) {
    rrect.size.height = rrect.size.width + rrect.size.height;
    rrect.size.width *= 2;
  } else {
    rrect.size.width = rrect.size.width + rrect.size.height;
    rrect.size.height *= 2;
  }

  *rotated_rect =rrect;
  return true;
                              
}

void OCDNetEngine::computeTextCandidatesMixNet(const cv::Mat& mask,
                                               std::vector<QUADANGLE>* const quads,
                                               std::vector<Text>* const texts, size_t* num_text,
                                               const ProcessParam& process_param) {
  Timer<100> timer;


  float resize_ratio = 0.4;
  cv::Mat resized;
  timer.Start();

  cv::resize(mask, resized,
             cv::Size(static_cast<int>(resize_ratio * mask.cols),
                      static_cast<int>(resize_ratio * mask.rows)));
  
  // cv::Mat element = cv::getStructuringElement(0, {7, 7});
  // cv::erode( resized, resized, element );
  // cv::dilate( resized, resized, element );

  std::vector<std::vector<cv::Point>> raw_contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(~resized, raw_contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
  // cv::imwrite("orig.png", ~resized);

  cv::Mat viz(resized.size(), CV_8UC3, cv::Scalar(0, 0, 0));
  cv::RNG rng(12345);

  std::vector<size_t> connected_compoment;
  std::vector<std::vector<size_t>> groups;
  for (size_t i = 0; i < hierarchy.size(); ++i) {
    auto const& h = hierarchy[i];
    connected_compoment.push_back(i);
    if (h[0] == -1 && h[2] == -1) {
      groups.push_back(connected_compoment);
      connected_compoment.clear();
      continue;
    }
    if (h[2] == -1 && h[3] == -1) {
      groups.push_back(connected_compoment);
      connected_compoment.clear();
      continue;
    }
  }
  LOG(INFO) << "find text group: " << groups.size();

// #pragma omp parallel for num_threads(8)
  for (const auto &group: groups) {
    cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

    // LOG(INFO) << "found on goup " <<i << " group size: " << group[i].size();

    cv::drawContours(viz, std::vector<std::vector<cv::Point>>{raw_contours[group[0]]}, -1,
                     color);

    if (group.size() == 1) {
      // no inner outer, 
    }

    // 0 always = outer, start from 1
    for (size_t j = 1; j < group.size(); ++j) {
      cv::RotatedRect ret;
      if (findMatryoshkaText(raw_contours[group[j]], raw_contours[group[0]], viz, color,
                             &ret, process_param.min_pixel_area)) {
        cv::drawContours(viz, std::vector<std::vector<cv::Point>>{raw_contours[group[j]]}, -1, color);
        auto& vertices = (*quads)[*num_text];
        Text& text = (*texts)[*num_text];

        ret.center = ret.center / resize_ratio;
        ret.size = ret.size / resize_ratio;


        ret.points(&vertices[0]);
        correctQuad(vertices);

#pragma unroll
        for (size_t k = 0; k < QUAD; ++k) {
          text.polygon[k* 2] = vertices[k].x;
          text.polygon[k * 2 + 1] = vertices[k].y;
        }
        *num_text += 1;

        // todo need to return with higher score
        if (*num_text == process_param.max_candidate) {
           return;
        }

      };

      // if (group[i][j] == inner) continue;
      // cv::drawContours(viz, std::vector<std::vector<cv::Point>>{raw_contours[group[i][j]]}, -1, color);

      // cv::Point p1(line[2] - 100 * line[0], line[3] - 100 * line[1]);

      // // cv::Point p1(line[2] , line[3]);
      // cv::Point p2(line[2] + 100 * line[0], line[3] + 100 * line[1]);

      // cv::line(viz, p1, p2, color);
      // break;
    }
    // auto rotated = cv::minAreaRect(raw_contours[outer]);

    // // prepare output
    // auto& vertices = (*quads)[*num_text];
    // Text& text = (*texts)[*num_text];

    // rotated.points(&vertices[0]);
    // correctQuad(vertices);

    // #pragma unroll
    //     for (size_t j = 0; j < QUAD; ++j) {
    //       text.polygon[j * 2] = vertices[j].x;
    //       text.polygon[j * 2 + 1] = vertices[j].y;
    //     }
    //     *num_text += 1;


  }
  LOG(ERROR) << "find contours: " << timer;

  cv::imwrite("hctnr.png", viz);
}

}  // namespace nvocdr

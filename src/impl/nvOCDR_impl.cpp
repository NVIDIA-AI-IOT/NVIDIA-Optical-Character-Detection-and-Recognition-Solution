#include "nvOCDR_impl.h"
#include <glog/logging.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include "nvocdr.h"

namespace nvocdr {

inline size_t getCnt(size_t input, size_t raw, size_t stride) {
  if (raw < input) {
    return 1;
  } else {
    return (raw - input) % stride == 0 ? (raw - input) / stride + 1 : (raw - input) / stride + 2;
  }
}

void nvOCDR::process(const nvOCDRInput& input, nvOCDROutput* const output) {
  mE2ETimer.Start();

  // restore image from buffer to cv::Mat, handle data order/channel
  restoreImage(input);
  // handle different strategy ,mResizeInfo will be set
  handleStrategy(input);
  // preprocess the data, like normalization
  preprocessInputImage();
  // all strategy will map to uniformed tile processing,
  processTile(input);
  // set the output
  setOutput(output);

  // wall time of processing
  mE2ETimer.Stop();
}

void nvOCDR::restoreImage(const nvOCDRInput& input) {

  if (input.data_format == DATAFORMAT_TYPE_HWC) {
    mInputImage = cv::Mat(cv::Size(input.width, input.height), CV_8UC(input.num_channel),
                          input.data, cv::Mat::AUTO_STEP);
  } else {
    // todo(shuohanc) restore from chw.
    LOG(ERROR) << "not implemented CHW";
    throw std::runtime_error("not implement CHW");
  }

  if (mInputImage.empty()) {
    throw std::runtime_error("input empty");
  }
}

void nvOCDR::handleStrategy(const nvOCDRInput& input) {
  const auto ocd_input_h = static_cast<int>(mOCDNet->getInputH());
  const auto ocd_input_w = static_cast<int>(mOCDNet->getInputW());

  float hw_ratio = static_cast<float>(input.height) / static_cast<float>(input.width);
  float h_ratio = static_cast<float>(input.height) / static_cast<float>(ocd_input_h);
  float w_ratio = static_cast<float>(input.width) / static_cast<float>(ocd_input_w);

  LOG_IF(ERROR, mParam.process_param.debug_log) << "hw (origin_h:origin_w) ratio: " << hw_ratio;
  LOG_IF(ERROR, mParam.process_param.debug_log) << "H (origin:model) ratio: " << h_ratio;
  LOG_IF(ERROR, mParam.process_param.debug_log) << "W (origin:model) ratio: " << w_ratio;

  if (mParam.process_param.strategy == STRATEGY_TYPE_SMART) {
    if (hw_ratio >= 0.95 && hw_ratio <= 1.05 && h_ratio >= 0.95 && h_ratio <= 1.05 &&
        w_ratio >= 0.95 && w_ratio <= 1.05) {  // approx square, also the dimension are close
      LOG_IF(ERROR, mParam.process_param.debug_log) << "resize input exact equal to model input";
      mResizeInfo = {
          {static_cast<int>(input.width), static_cast<int>(input.height)},
          {ocd_input_w, ocd_input_h},
      };
    } else if (hw_ratio >= 0.95 && hw_ratio <= 1.06) {  // approx square, but dimension diff a lot
      mResizeInfo = {
          {static_cast<int>(input.width), static_cast<int>(input.height)},
          {static_cast<int>(input.width), static_cast<int>(input.height)},
      };
    } else {
      mResizeInfo = {
          {static_cast<int>(input.width), static_cast<int>(input.height)},
          {static_cast<int>(input.width), static_cast<int>(input.height)},
      };
    }
  } else if (mParam.process_param.strategy == STRATEGY_TYPE_RESIZE_TILE) {
    // resize short to
    if (hw_ratio < 1) {  // h = short
      mResizeInfo = {
          {static_cast<int>(input.width), static_cast<int>(input.height)},
          {static_cast<int>(ocd_input_h / hw_ratio), ocd_input_h},
      };
    } else {  // w = short
      mResizeInfo = {
          {static_cast<int>(input.width), static_cast<int>(input.height)},
          {ocd_input_w, static_cast<int>(ocd_input_w * hw_ratio)},
      };
    }

  } else if (mParam.process_param.strategy == STRATEGY_TYPE_NORESIZE_TILE) {
    mResizeInfo = {
        {static_cast<int>(input.width), static_cast<int>(input.height)},
        {static_cast<int>(input.width), static_cast<int>(input.height)},
    };
  }

  if (mResizeInfo.first == mResizeInfo.second) {
    LOG(INFO) << "no resize: " << mResizeInfo.first;
  } else {
    LOG(INFO) << "resize: " << mResizeInfo.first << " --> " << mResizeInfo.second;
  }

  cv::resize(mInputImage, mInputImageResized, mResizeInfo.second);

  // resize image to destinate size
  mOCDScoreMap = cv::Mat(mResizeInfo.second, CV_32F, cv::Scalar(0.F));
  mOCDValidCntMap = cv::Mat(mResizeInfo.second, CV_32F, cv::Scalar(0.F));
}

void nvOCDR::getTilePlan(size_t input_w, size_t input_h, size_t raw_w, size_t raw_h,
                         size_t stride) {
  LOG_IF(ERROR, mParam.process_param.debug_log)
      << input_w << " " << input_h << " " << raw_w << " " << raw_h << " " << stride << "\n";
  size_t h_cnt = getCnt(input_h, raw_h, stride);
  size_t w_cnt = getCnt(input_w, raw_w, stride);

  LOG_IF(ERROR, mParam.process_param.debug_log)
      << "plan h steps: " << h_cnt << ", w steps " << w_cnt;

  mTiles.clear();
  for (size_t i = 0; i < h_cnt; ++i) {
    for (size_t j = 0; j < w_cnt; ++j) {
      // all tile guarenteed to be inside the image
      cv::Point br(std::min(j * stride + input_w, raw_w), std::min(i * stride + input_h, raw_h));
      cv::Point tl(std::max(br.x - static_cast<int>(input_w), 0),
                   std::max(br.y - static_cast<int>(input_h), 0));
      mTiles.emplace_back(tl, br);
      LOG_IF(ERROR, mParam.process_param.debug_log) << "tile " << mTiles.back();
    }
  }
}


void nvOCDR::setOutput(nvOCDROutput* const output) {
  output->num_texts = mNumTexts;
  output->texts = &mTexts[0];
}

void nvOCDR::preprocessInputImage() {
  static cv::Scalar BGR_MEAN(IMG_MEAN_B, IMG_MEAN_G, IMG_MEAN_R);
  static cv::Scalar BGR_SCALE(IMG_SCALE_BRG, IMG_SCALE_BRG, IMG_SCALE_BRG);

  static cv::Scalar GRAY_MEAN(IMG_MEAN_GRAY);
  static cv::Scalar GRAY_SCALE(IMG_SCALE_GRAY);

  // ocd input preprocess, bgr normalize
  mInputImageResized.convertTo(mInputImageResized32F, CV_32FC3);
  // normalize
  mInputImageResized32F -= BGR_MEAN;
  mInputImageResized32F = mInputImageResized32F.mul(BGR_SCALE);

  // ocr input preprocess, gray normalize
  cv::cvtColor(mInputImage, mInputGrayImage, cv::COLOR_BGR2GRAY);
  mInputGrayImage.convertTo(mInputGrayImage, CV_32F);
  mInputGrayImage -= GRAY_MEAN;
  mInputGrayImage = mInputGrayImage.mul(GRAY_SCALE);
};

void nvOCDR::processTile(const nvOCDRInput& input) {
  const auto ocd_input_h = mOCDNet->getInputH();
  const auto ocd_input_w = mOCDNet->getInputW();

  // parameterize stride ??
  getTilePlan(ocd_input_w, ocd_input_h, mInputImageResized.cols, mInputImageResized.rows,
              static_cast<size_t>(ocd_input_h * 0.9));

  size_t num_tiles = mTiles.size();
  size_t num_ocd_bs = mOCDNet->getBatchSize();
  size_t num_ocd_runs =
      num_tiles % num_ocd_bs == 0 ? num_tiles / num_ocd_bs : num_tiles / num_ocd_bs + 1;
  LOG(INFO) << "tiles: " << num_tiles << ", run ocd " << num_ocd_runs << " times with batch size "
            << num_ocd_bs;

  mOCDTimer.Start();
  // 1. ocd process
  for (size_t i = 0; i < num_ocd_runs; i++) {
    size_t start_idx = i * num_ocd_bs;
    size_t end_idx = std::min((i + 1) * num_ocd_bs, num_tiles);
    auto t = std::chrono::high_resolution_clock::now();
    preprocessOCDTile(start_idx, end_idx);  // batch inference
    // ocd_dur += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t);

    mOCDNet->syncMemory(true, true, mStream);
    mOCDNet->infer();
    mOCDNet->syncMemory(false, false, mStream);
    cudaStreamSynchronize(mStream);
    postprocessOCDTile(start_idx, end_idx);  // restore to image

    cv::inRange(mOCDScoreMap / mOCDValidCntMap, mParam.process_param.binarize_lower_threshold,
                mParam.process_param.binarize_upper_threshold, mOCDOutputMask);

    mOCDOutputMask = ~mOCDOutputMask;
    // restore mask to origin size
    cv::resize(mOCDOutputMask, mOCDOutputMask, mResizeInfo.first);
  }
  // mOCDTimer.Stop();
  LOG(INFO) << "ocd process time: " << mOCDTimer;

  if (mParam.process_param.debug_image) {
    cv::imwrite("text_area.png", ~mOCDOutputMask);
  }

  mSelectProcessTimer.Start();
  // select and filter the proper text candidates
  selectOCRCandidates();
  LOG(INFO) << "select text process time: " << mSelectProcessTimer;

  // 2. ocr process
  size_t num_ocr_bs = mOCRNet->getBatchSize();
  size_t num_ocr_runs =
      mNumTexts % num_ocr_bs == 0 ? mNumTexts / num_ocr_bs : mNumTexts / num_ocr_bs + 1;
  LOG(INFO) << "found text area number: " << mNumTexts << ", run ocr " << num_ocr_runs
            << " times with batch size " << num_ocr_bs;

  std::vector<size_t> directions;
  if (mParam.process_param.all_direction) {
    directions = {0, 1, 2, 3};
  } else {
    directions = {0};
  }

  mOCRTimer.Start();

  for (size_t i = 0; i < num_ocr_runs; ++i) {
    size_t start_idx = i * num_ocr_bs;
    size_t end_idx = std::min((i + 1) * num_ocr_bs, mNumTexts);
    for (auto const& bl_idx : directions) {
      preprocessOCR(start_idx, end_idx, bl_idx);

      mOCRNet->syncMemory(true, true, mStream);
      mOCRNet->infer();
      mOCRNet->syncMemory(false, false, mStream);
      cudaStreamSynchronize(mStream);

      postprocessOCR(start_idx, end_idx);
    }
  }

  LOG(INFO) << "ocr process takes " << mOCRTimer;
}

void nvOCDR::postprocessOCDTile(size_t start, size_t end) {
  float* output_buf = mOCDNet->getMaskOutputBuf();

  // !!! todo(shuohanc) use input size, cause they are same for now
  auto const output_h = mOCDNet->getInputH();
  auto const output_w = mOCDNet->getInputW();
  for (size_t i = start; i < end; ++i) {
    const auto& tile = mTiles[i];
    cv::Mat score(output_h, output_w, CV_32F,
                  output_buf + mOCDNet->getOutputChannelIdx() * output_h * output_w,
                  cv::Mat::AUTO_STEP);
    mOCDScoreMap(tile) += score(cv::Rect(cv::Point(0, 0), tile.br() - tile.tl()));

    mOCDValidCntMap(tile) += 1;
    output_buf += mOCDNet->getOutputChannels() * output_h * output_w;

    if (mParam.process_param.debug_image) {
      cv::imwrite("tile_" + std::to_string(i) + ".png", score * 200);
    }
  }
}

void nvOCDR::preprocessOCDTile(size_t start, size_t end) {
  float* buf = reinterpret_cast<float*>(
      mBufManager.getBuffer(mOCDNet->getBufName(OCDNET_INPUT), BUFFER_TYPE::HOST));

  const auto ocd_input_h = mOCDNet->getInputH();
  const auto ocd_input_w = mOCDNet->getInputW();

  // fill ocd batch, and normalize
  for (size_t j = start; j < end; j++) {
    const auto& tile = mTiles[j];
    // 1 tile -> 1 batch in OCD
    cv::Mat ocd_batch(cv::Size(ocd_input_w, ocd_input_h), CV_32FC3, cv::Scalar(0, 0, 0));
    // LOG(INFO) << mInputImageResized32F.size() << "\t" << tile ;
    mInputImageResized32F(tile).copyTo(ocd_batch(cv::Rect(cv::Point(0, 0), tile.br() - tile.tl())));

    std::vector<cv::Mat> dummy_channels;
#pragma unroll
    for (size_t c = 0; c < 3; ++c) {
      dummy_channels.emplace_back(ocd_input_h, ocd_input_w, CV_32F, buf);
      buf += ocd_input_h * ocd_input_w;
    }
    cv::split(ocd_batch, dummy_channels);
  }
}

void nvOCDR::selectOCRCandidates() {
  mNumTexts = 0;
  mOCDNet->computeTextCandidates(mOCDOutputMask, &mQuadPts, &mTexts, &mNumTexts,
                                 mParam.process_param);
}

void nvOCDR::preprocessOCR(size_t start, size_t end, size_t bl_pt_idx) {
  // fill the ocr input buffer
  const auto ocr_input_h = mOCRNet->getInputH();
  const auto ocr_input_w = mOCRNet->getInputW();

  static QUADANGLE transform_dst{
      cv::Point2f{0.F, static_cast<float>(ocr_input_h)}, cv::Point2f{0.F, 0.F},
      cv::Point2f{static_cast<float>(ocr_input_w - 1), 0.F},
      cv::Point2f{static_cast<float>(ocr_input_w - 1), static_cast<float>(ocr_input_h - 1)}};
  static QUADANGLE transform_src;

  float* buf = reinterpret_cast<float*>(
      mBufManager.getBuffer(mOCRNet->getBufName(OCRNET_INPUT), BUFFER_TYPE::HOST));
  for (size_t i = start; i < end; ++i) {
    auto const& quad_pts = mQuadPts[i];
    cv::Rect rect = cv::boundingRect(quad_pts);
#pragma unroll
    for (size_t j = 0; j < QUAD; ++j) {
      transform_src[j] =
          quad_pts[(j + bl_pt_idx) % QUAD] -
          cv::Point2f{static_cast<float>(rect.tl().x), static_cast<float>(rect.tl().y)};
    }

    // compute perspective transform
    cv::Mat h = cv::findHomography(transform_src, transform_dst);
    cv::Mat text_roi(ocr_input_h, ocr_input_w, CV_32F, buf);

    if ((rect & cv::Rect(0, 0, mInputGrayImage.cols, mInputGrayImage.rows)) ==
        rect) {  // rect not image
      auto tl = rect.tl();
      auto br = rect.br();
      tl.x = std::max(tl.x, 0);
      tl.y = std::max(tl.y, 0);
      br.x = std::min(br.x, mInputGrayImage.cols);
      br.y = std::min(br.y, mInputGrayImage.rows);
    } else {
      continue;
      LOG(WARNING) << "ignore text out side ";
    }

    const auto start_t = std::chrono::high_resolution_clock::now();

    cv::warpPerspective(mInputGrayImage(rect), text_roi, h, cv::Size(ocr_input_w, ocr_input_h));
    const auto end_t = std::chrono::high_resolution_clock::now();

    buf += ocr_input_h * ocr_input_w;

    // for debug
    if (mParam.process_param.debug_log) {
      cv::imwrite("text_" + std::to_string(i) + "_" + std::to_string(bl_pt_idx) + ".png",
                  denormalizeGray(text_roi));
    }
  }
}

cv::Mat nvOCDR::denormalizeGray(const cv::Mat& input) {
  cv::Mat ret;
  input.copyTo(ret);
  static cv::Scalar GRAY_MEAN(IMG_MEAN_GRAY);
  static cv::Scalar GRAY_SCALE(IMG_SCALE_GRAY);
  ret /= GRAY_SCALE;
  ret += GRAY_MEAN;
  return ret;
}

void nvOCDR::postprocessOCR(size_t start, size_t end) {
  for (size_t i = start; i < end; ++i) {
    Text& text = mTexts[i];
    mOCRNet->decode(&text, i - start);
  }
}

nvOCDR::nvOCDR(const nvOCDRParam& param) : mParam(param) {
  // todo(shuohanc) check param
  mTexts.resize(param.process_param.max_candidate);
  mQuadPts.resize(param.process_param.max_candidate);
  cudaStreamCreate(&mStream);

  LOG(INFO) << "================= init ocr =================";
  mOCRNet = std::make_unique<OCRNetEngine>(OCR_PREFIX, param.ocr_param);
  mOCRNet->init();

  LOG(INFO) << "================= init ocd =================";
  mOCDNet = std::make_unique<OCDNetEngine>(OCD_PREFIX, param.ocd_param);
  mOCDNet->init();

  // warmp up
  for (size_t i = 0; i < NUM_WARMUP_RUNS; ++i) {
    mOCRNet->infer(mStream);
    mOCDNet->infer(mStream);
  }
}

void nvOCDR::printTimeStat() {
  LOG(INFO) << "---------- time statistics ----------";
  LOG(INFO) << "history size: " << TIME_HISTORY_SIZE;

  auto ocd_mean = mOCDTimer.getMean();
  auto select_mean = mSelectProcessTimer.getMean();
  auto ocr_mean = mOCRTimer.getMean();
  auto e2e_mean = mE2ETimer.getMean();
  LOG(INFO) << "statistic shows in 'mean(ms) / percentage(%)' ";

  LOG(INFO) << "ocd: " << ocd_mean << "ms / " << static_cast<int>(ocd_mean / e2e_mean * 100) << "%"; 
  LOG(INFO) << "ocr: " << ocr_mean << "ms / " << static_cast<int>(ocr_mean / e2e_mean * 100) << "%"; 

  LOG(INFO) << "e2e: " << e2e_mean <<"ms"; 
}

}  // namespace nvocdr

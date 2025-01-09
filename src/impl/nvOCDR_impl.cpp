#include "nvOCDR_impl.h"
#include <glog/logging.h>
#include <nppi.h>
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

inline cv::Mat denormalizeGray(const cv::Mat& input, const COLOR_PREPROC_PARAM& param) {
  cv::Mat ret;
  input.copyTo(ret);
  ret /= param.rgb_scale;
  ret += param.r_mean;
  return ret;
}

inline cv::Mat denormalizeRGB(const cv::Mat& input, const COLOR_PREPROC_PARAM& param) {
  cv::Mat ret;
  input.copyTo(ret);
  ret = ret.mul(cv::Scalar(param.r_std, param.g_std, param.b_std));
  ret += cv::Scalar(param.r_mean, param.g_mean, param.b_mean);
  ret = ret * param.rgb_scale;
  return ret;
}

void nvOCDR::process(const nvOCDRInput& input, nvOCDROutput* const output) {
  mE2ETimer.Start();
  // restore image from buffer to cv::Mat, handle data order/channel
  restoreImage(input);
  // all strategy will map to uniformed tile processing,
  processTile(input);
  // set the output
  setOutput(output);
  // e2e time of processing
  mE2ETimer.Stop();
}

void nvOCDR::restoreImage(const nvOCDRInput& input) {

  if (input.data_format == DATAFORMAT_TYPE_HWC) {
    mInputImage = cv::Mat(mRawInputSize,  CV_8UC3, input.data, cv::Mat::AUTO_STEP);
    memcpy(mBufManager.getBuffer(ORIGIN_INPUT_BUF, HOST), mInputImage.data,
           mInputImage.total() * mInputImage.elemSize());
    mBufManager.copyHostToDevice(ORIGIN_INPUT_BUF, mStream);

  } else {
    // todo(shuohanc) restore from chw.
    LOG(ERROR) << "not implemented CHW";
    throw std::runtime_error("not implement CHW");
  }

  if (mInputImage.empty()) {
    throw std::runtime_error("input empty");
  }
}

void nvOCDR::handleStrategy() {
  const auto ocd_input_h = mOCDInputSize.height;
  const auto ocd_input_w = mOCDInputSize.width;

  float hw_ratio = static_cast<float>(mRawInputSize.height) / static_cast<float>(mRawInputSize.width);
  float h_ratio = static_cast<float>(mRawInputSize.height) / static_cast<float>(ocd_input_h);
  float w_ratio = static_cast<float>(mRawInputSize.width) / static_cast<float>(ocd_input_w);

  LOG_IF(ERROR, mParam.process_param.debug_log) << "hw (origin_h:origin_w) ratio: " << hw_ratio;
  LOG_IF(ERROR, mParam.process_param.debug_log) << "H (origin:model) ratio: " << h_ratio;
  LOG_IF(ERROR, mParam.process_param.debug_log) << "W (origin:model) ratio: " << w_ratio;

  if (mParam.process_param.strategy == STRATEGY_TYPE_SMART) {
    if (hw_ratio >= 0.95 && hw_ratio <= 1.05 && h_ratio >= 0.95 && h_ratio <= 1.05 &&
        w_ratio >= 0.95 && w_ratio <= 1.05) {  // approx square, also the dimension are close
      LOG_IF(ERROR, mParam.process_param.debug_log) << "resize input exact equal to model input";
      // mResizeInfo = {
      //     {static_cast<int>(mInputShape[W_IDX]), static_cast<int>(mInputShape[H_IDX])},
      //     {ocd_input_w, ocd_input_h},
      // };
      mDestinateSize = {ocd_input_w, ocd_input_h};
    } else if (hw_ratio >= 0.95 && hw_ratio <= 1.06) {  // approx square, but dimension diff a lot
      // mResizeInfo = {
      //     {static_cast<int>(mInputShape[W_IDX]), static_cast<int>(mInputShape[H_IDX])},
      //     {static_cast<int>(mInputShape[W_IDX]), static_cast<int>(mInputShape[H_IDX])},
      // };
      mDestinateSize = mRawInputSize;
    } else {
      // mResizeInfo = {
      //     {static_cast<int>(mInputShape[W_IDX]), static_cast<int>(mInputShape[H_IDX])},
      //     {static_cast<int>(mInputShape[W_IDX]), static_cast<int>(mInputShape[H_IDX])},
      // };
       mDestinateSize = mRawInputSize;
    }
  } else if (mParam.process_param.strategy == STRATEGY_TYPE_RESIZE_TILE) {
    // resize short to
    if (hw_ratio < 1) {  // h = short
      // mResizeInfo = {
      //     {static_cast<int>(mInputShape[W_IDX]), static_cast<int>(mInputShape[H_IDX])},
      //     {static_cast<int>(ocd_input_h / hw_ratio), ocd_input_h},
      // };
       mDestinateSize = {static_cast<int>(ocd_input_h / hw_ratio), ocd_input_h};
    } else {  // w = short
      // mResizeInfo = {
      //     {static_cast<int>(mInputShape[W_IDX]), static_cast<int>(mInputShape[H_IDX])},
      //     {ocd_input_w, static_cast<int>(ocd_input_w * hw_ratio)},
      // };
      mDestinateSize = {ocd_input_w, static_cast<int>(ocd_input_w * hw_ratio)};
    }

  } else if (mParam.process_param.strategy == STRATEGY_TYPE_NORESIZE_TILE) {
    // mResizeInfo = {
    //     {static_cast<int>(mInputShape[W_IDX]), static_cast<int>(mInputShape[H_IDX])},
    //     {static_cast<int>(mInputShape[W_IDX]), static_cast<int>(mInputShape[H_IDX])},
    // };
    mDestinateSize = mRawInputSize;
  } else if (mParam.process_param.strategy == STRATEGY_TYPE_RESIZE_FULL) {
    // mResizeInfo = {
    //     {static_cast<int>(mInputShape[W_IDX]), static_cast<int>(mInputShape[H_IDX])},
    //     {static_cast<int>(ocd_input_w), static_cast<int>(ocd_input_h)},
    // };
    mDestinateSize = {ocd_input_w, ocd_input_h};
  }

  // parameterize stride ?? use .95 temporarily
  getTilePlan(ocd_input_w, ocd_input_h, mDestinateSize.width, mDestinateSize.height,
              static_cast<size_t>(ocd_input_h * 0.95));
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

void nvOCDR::initBuffers() {
  mBufManager.initBuffer(ORIGIN_INPUT_BUF,
                         mInputChannels * mRawInputSize.width * mRawInputSize.height,
                         true);                                             
}

// main control process 
void nvOCDR::processTile(const nvOCDRInput& input) {
  const auto ocd_input_h = mOCDInputSize.height;
  const auto ocd_input_w = mOCDInputSize.width;

  size_t num_tiles = mTiles.size();
  size_t num_ocd_bs = mOCDProcessor->getBatchSize();
  size_t num_ocd_runs =
      num_tiles % num_ocd_bs == 0 ? num_tiles / num_ocd_bs : num_tiles / num_ocd_bs + 1;
  LOG(INFO) << "tiles: " << num_tiles << ", run ocd " << num_ocd_runs << " times with batch size "
            << num_ocd_bs;

  mOCDTimer.Start();

  // resize image to destinate size
  mOCDScoreMap = cv::Mat::zeros(mDestinateSize, CV_32F);
  mOCDValidCntMap = cv::Mat::zeros(mDestinateSize, CV_32F);

  // 1. ocd process
  for (size_t i = 0; i < num_ocd_runs; i++) {
    size_t start_idx = i * num_ocd_bs;
    size_t end_idx = std::min((i + 1) * num_ocd_bs, num_tiles);
    auto t = std::chrono::high_resolution_clock::now();
    preprocessOCDTileGPU(start_idx, end_idx);

    mOCDProcessor->infer(false, mStream);
    cudaStreamSynchronize(mStream);
    postprocessOCDTile(start_idx, end_idx);
  }
  LOG(INFO) << "ocd process time: " << mOCDTimer;

  // 2. select and filter the proper text area candidates
  selectOCRCandidates();

  // 3. ocr process
  mOCRTimer.Start();
  size_t num_ocr_bs = mOCRProcessor->getBatchSize();
  size_t num_ocr_runs =
      mNumTexts % num_ocr_bs == 0 ? mNumTexts / num_ocr_bs : mNumTexts / num_ocr_bs + 1;
  LOG(INFO) << "found text area number: " << mNumTexts << ", run ocr " << num_ocr_runs
            << " times with batch size " << num_ocr_bs;

  for (size_t i = 0; i < num_ocr_runs; ++i) {
    size_t start_idx = i * num_ocr_bs;
    size_t end_idx = std::min((i + 1) * num_ocr_bs, mNumTexts);
    for (auto const& bl_idx : mOCRDirections) {
      preprocessOCR(start_idx, end_idx, bl_idx);
      mOCRProcessor->infer(false, mStream);
      cudaStreamSynchronize(mStream);
      postprocessOCR(start_idx, end_idx);
    }
  }
  LOG(INFO) << "ocr process takes " << mOCRTimer;
}

void nvOCDR::postprocessOCDTile(size_t start, size_t end) {
  float* output_buf = mOCDProcessor->getMaskOutputBuf();

  // !!! todo(shuohanc) use input size, cause they are same for now
  const auto output_h = mOCDInputSize.height;
  const auto output_w = mOCDInputSize.width;
  for (size_t i = start; i < end; ++i) {
    const auto& tile = mTiles[i];
    cv::Mat score(output_h, output_w, CV_32F,
                  output_buf + mOCDProcessor->getOutputChannelIdx() * output_h * output_w,
                  cv::Mat::AUTO_STEP);
    mOCDScoreMap(tile) += score(cv::Rect(cv::Point(0, 0), tile.br() - tile.tl()));

    mOCDValidCntMap(tile) += 1;
    output_buf += mOCDProcessor->getOutputChannels() * output_h * output_w;

    if (mParam.process_param.debug_image) {
      cv::imwrite("tile_" + std::to_string(i) + ".png", score * 200);
    }
  }
}

void nvOCDR::preprocessOCDTileGPU(size_t start, size_t end) {
  cv::Size2f scale(static_cast<float>(mRawInputSize.width) / mDestinateSize.width,
                   static_cast<float>(mRawInputSize.height) / mDestinateSize.height);
  const auto output_h = mOCDInputSize.height;
  const auto output_w = mOCDInputSize.width;

  cv::Mat3d h_inv = cv::Mat::eye(3, 3, CV_64F);

  h_inv.at<double>(0,0) = scale.width;
  h_inv.at<double>(1,1) = scale.height;
  for (size_t i = start; i < end; i++) {
    const auto& tile = mTiles[i];

    h_inv.at<double>(0,2) = tile.x * scale.width;
    h_inv.at<double>(1,2) = tile.y * scale.height;

    if (mParam.ocd_param.type == nvOCDParam::OCD_MODEL_TYPE::OCD_MODEL_TYPE_NORMAL) {
      launch_fused_warp_perspective<true, true, false>(
          static_cast<uint8_t*>(mBufManager.getBuffer(ORIGIN_INPUT_BUF, DEVICE)),
          static_cast<float*>(mBufManager.getBuffer(mOCDProcessor->getInputBufName(), DEVICE)) + 3 * output_w * output_h * (i - start),
          mRawInputSize, tile.size(),
          OCD_NORMAL_PREPROR_PARAM, mStream, h_inv);
    } else if (mParam.ocd_param.type == nvOCDParam::OCD_MODEL_TYPE::OCD_MODEL_TYPE_MIXNET) {
      launch_fused_warp_perspective<true, false, false>(
          static_cast<uint8_t*>(mBufManager.getBuffer(ORIGIN_INPUT_BUF, DEVICE)),
          static_cast<float*>(mBufManager.getBuffer(mOCDProcessor->getInputBufName(), DEVICE)) + 3 * output_w * output_h * (i - start),
          mRawInputSize, tile.size(),
          OCD_MIXNET_PREPROR_PARAM, mStream, h_inv);
    }
  }
}

void nvOCDR::selectOCRCandidates() {
  mSelectProcessTimer.Start();
  mNumTexts = 0;
  cv::inRange(mOCDScoreMap, mParam.process_param.binarize_lower_threshold,
              mParam.process_param.binarize_upper_threshold, mOCDOutputMask);
  cv::resize(~mOCDOutputMask, mOCDOutputMask, mRawInputSize);
  if (mParam.process_param.debug_image) {
    cv::imwrite("text_area.png", ~mOCDOutputMask);
  }
  // the selection logic is high related to ocd model, move the process into ocd processor
  mOCDProcessor->computeTextCandidates(mOCDOutputMask, &mQuadPts, &mTexts, &mNumTexts,
                                       mParam.process_param);
  LOG(INFO) << "select text process time: " << mSelectProcessTimer;
}

void nvOCDR::preprocessOCR(size_t start, size_t end, size_t bl_pt_idx) {
  // fill the ocr input buffer
  const auto ocr_input_h = mOCRInputSize.height;
  const auto ocr_input_w = mOCRInputSize.width;

  static QUADANGLE transform_dst{
      cv::Point2f{0.F, static_cast<float>(ocr_input_h)}, cv::Point2f{0.F, 0.F},
      cv::Point2f{static_cast<float>(ocr_input_w - 1), 0.F},
      cv::Point2f{static_cast<float>(ocr_input_w - 1), static_cast<float>(ocr_input_h - 1)}};
  static QUADANGLE transform_src;

  for (size_t i = start; i < end; ++i) {
    auto const& quad_pts = mQuadPts[i];
#pragma unroll
    for (size_t j = 0; j < QUAD; ++j) {
      transform_src[j] = quad_pts[(j + bl_pt_idx) % QUAD];
    }
    // compute perspective transform
    cv::Mat h_inv = cv::findHomography(transform_src, transform_dst).inv();
    if (mParam.ocr_param.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP) {
      launch_fused_warp_perspective<true, false, false>(
          static_cast<uint8_t*>(mBufManager.getBuffer(ORIGIN_INPUT_BUF, DEVICE)),
          static_cast<float*>(mBufManager.getBuffer(mOCRProcessor->getInputBufName(), DEVICE)) + 3 * ocr_input_w * ocr_input_h * (i - start),
          mRawInputSize, {ocr_input_w, ocr_input_h},
          OCD_NORMAL_PREPROR_PARAM, mStream, h_inv);
    } else {
      launch_fused_warp_perspective<true, false, true>(
          static_cast<uint8_t*>(mBufManager.getBuffer(ORIGIN_INPUT_BUF, DEVICE)),
          static_cast<float*>(mBufManager.getBuffer(mOCRProcessor->getInputBufName(), DEVICE)) + ocr_input_w * ocr_input_h * (i - start),
          mRawInputSize, {ocr_input_w, ocr_input_h},
          OCR_PREPROC_GRAY_PARAM, mStream, h_inv);
      }
  }
  
  if (mParam.process_param.debug_image) {
    mBufManager.copyDeviceToHost(mOCRProcessor->getInputBufName(), mStream);
    float* buff = (float*)mBufManager.getBuffer(mOCRProcessor->getInputBufName(), HOST);

    for (size_t i = start; i < end; ++i) {
      if (mParam.ocr_param.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP) {
        cv::Mat r(ocr_input_h, ocr_input_w, CV_32F, buff);
        cv::Mat g(ocr_input_h, ocr_input_w, CV_32F, buff + 1 * ocr_input_w * ocr_input_h);
        cv::Mat b(ocr_input_h, ocr_input_w, CV_32F, buff + 2 * ocr_input_w * ocr_input_h);
        cv::Mat debug_text;
        cv::merge(std::vector<cv::Mat>{r, g, b}, debug_text);
        buff += 3 * ocr_input_w * ocr_input_h;
        
        cv::imwrite("text_" + std::to_string(i) + "_" + std::to_string(bl_pt_idx) + ".png",
                    denormalizeRGB(debug_text, OCD_NORMAL_PREPROR_PARAM));
      } else {
        cv::Mat debug_text(ocr_input_h, ocr_input_w, CV_32F, buff + ocr_input_w * ocr_input_h * (i - start));
        cv::imwrite("text_" + std::to_string(i) + "_" + std::to_string(bl_pt_idx) + ".png",
                    denormalizeGray(debug_text, OCR_PREPROC_GRAY_PARAM));
      }
    }
  }
}



void nvOCDR::postprocessOCR(size_t start, size_t end) {
  for (size_t i = start; i < end; ++i) {
    mOCRProcessor->decode(&mTexts[i], i - start);
  }
}

nvOCDR::nvOCDR(const nvOCDRParam& param) : mParam(param) {
  // todo(shuohanc) check param
  if (param.input_shape[0] != 3 || param.input_shape[1] == 0 || param.input_shape[2] == 0) {
    throw std::runtime_error("input shape error");
  }

  cudaStreamCreate(&mStream);

  // memcpy(mInputShape.data(), param.input_shape, mInputShape.size() * sizeof(size_t));
  mInputChannels = param.input_shape[C_IDX];
  mRawInputSize.width = param.input_shape[W_IDX];
  mRawInputSize.height = param.input_shape[H_IDX];

  LOG(INFO) << "input shape set to WxH: " << mRawInputSize;
  initBuffers();
  mTexts.resize(param.process_param.max_candidate);
  mQuadPts.resize(param.process_param.max_candidate);

  LOG(INFO) << "================= init ocr =================";
  mOCRProcessor = std::make_unique<OCRProcessor>(param.ocr_param);
  // mOCRNet = std::make_unique<OCRNetEngine>(OCR_PREFIX, param.ocr_param);
  mOCRProcessor->init();

  LOG(INFO) << "================= init ocd =================";
  mOCDProcessor = std::make_unique<OCDProcessor>(param.ocd_param);
  mOCDProcessor->init();

  // mOCDNet = std::make_unique<OCDNetEngine>(OCD_PREFIX, param.ocd_param);
  mOCDInputSize = mOCDProcessor->getInputHW();
  mOCRInputSize = mOCRProcessor->getInputHW();

  LOG(INFO) << "ocd model with input WXH: " << mOCDInputSize;
  LOG(INFO) << "ocr model with input WXH: " << mOCRInputSize;
  // warmp up
  for (size_t i = 0; i < NUM_WARMUP_RUNS; ++i) {
    // mOCDProcessor->infer(true, mStream);
    // mOCRProcessor->infer(true, mStream);
  }
  if (mParam.process_param.all_direction) {
    // recog on all specified directions
    mOCRDirections = {0, 1, 2, 3};
  } else {
    mOCRDirections = {0};
  }

  // handle different strategy ,mResizeInfo will be set
  handleStrategy();

  if (mRawInputSize == mDestinateSize) {
    LOG(INFO) << "no resize: " << mRawInputSize;
  } else {
    LOG(INFO) << "resize: " << mRawInputSize << " --> " << mDestinateSize;
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
  LOG(INFO) << "selector: " << select_mean << "ms / "
            << static_cast<int>(select_mean / e2e_mean * 100) << "%";
  LOG(INFO) << "ocr: " << ocr_mean << "ms / " << static_cast<int>(ocr_mean / e2e_mean * 100) << "%";

  LOG(INFO) << "e2e: " << e2e_mean << "ms";
}

}  // namespace nvocdr

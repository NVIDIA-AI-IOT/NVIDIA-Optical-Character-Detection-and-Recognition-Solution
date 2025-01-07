#include "nvOCDR_impl.h"
#include <glog/logging.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include "nvocdr.h"
#include <nppi.h>
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

  // LOG(INFO) << "tmp timer: " << mTmpTimer.getMean();

  // wall time of processing
  mE2ETimer.Stop();
}

void nvOCDR::restoreImage(const nvOCDRInput& input) {

  if (input.data_format == DATAFORMAT_TYPE_HWC) {
    mInputImage = cv::Mat(cv::Size(mInputShape[W_IDX], mInputShape[H_IDX]), CV_8UC3,
                          input.data, cv::Mat::AUTO_STEP);
    memcpy(mBufManager.getBuffer(ORIGIN_INPUT_BUF, HOST), mInputImage.data, mInputImage.total() * mInputImage.elemSize());
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

void nvOCDR::handleStrategy(const nvOCDRInput& input) {
  const auto ocd_input_h = mOCDInputSize.height;
  const auto ocd_input_w = mOCDInputSize.width;

  float hw_ratio = static_cast<float>(mInputShape[H_IDX]) / static_cast<float>(mInputShape[W_IDX]);
  float h_ratio = static_cast<float>(mInputShape[H_IDX]) / static_cast<float>(ocd_input_h);
  float w_ratio = static_cast<float>(mInputShape[W_IDX]) / static_cast<float>(ocd_input_w);

  LOG_IF(ERROR, mParam.process_param.debug_log) << "hw (origin_h:origin_w) ratio: " << hw_ratio;
  LOG_IF(ERROR, mParam.process_param.debug_log) << "H (origin:model) ratio: " << h_ratio;
  LOG_IF(ERROR, mParam.process_param.debug_log) << "W (origin:model) ratio: " << w_ratio;

  if (mParam.process_param.strategy == STRATEGY_TYPE_SMART) {
    if (hw_ratio >= 0.95 && hw_ratio <= 1.05 && h_ratio >= 0.95 && h_ratio <= 1.05 &&
        w_ratio >= 0.95 && w_ratio <= 1.05) {  // approx square, also the dimension are close
      LOG_IF(ERROR, mParam.process_param.debug_log) << "resize input exact equal to model input";
      mResizeInfo = {
          {static_cast<int>(mInputShape[W_IDX]), static_cast<int>(mInputShape[H_IDX])},
          {ocd_input_w, ocd_input_h},
      };
    } else if (hw_ratio >= 0.95 && hw_ratio <= 1.06) {  // approx square, but dimension diff a lot
      mResizeInfo = {
          {static_cast<int>(mInputShape[W_IDX]), static_cast<int>(mInputShape[H_IDX])},
          {static_cast<int>(mInputShape[W_IDX]), static_cast<int>(mInputShape[H_IDX])},
      };
    } else {
      mResizeInfo = {
          {static_cast<int>(mInputShape[W_IDX]), static_cast<int>(mInputShape[H_IDX])},
          {static_cast<int>(mInputShape[W_IDX]), static_cast<int>(mInputShape[H_IDX])},
      };
    }
  } else if (mParam.process_param.strategy == STRATEGY_TYPE_RESIZE_TILE) {
    // resize short to
    if (hw_ratio < 1) {  // h = short
      mResizeInfo = {
          {static_cast<int>(mInputShape[W_IDX]), static_cast<int>(mInputShape[H_IDX])},
          {static_cast<int>(ocd_input_h / hw_ratio), ocd_input_h},
      };
    } else {  // w = short
      mResizeInfo = {
          {static_cast<int>(mInputShape[W_IDX]), static_cast<int>(mInputShape[H_IDX])},
          {ocd_input_w, static_cast<int>(ocd_input_w * hw_ratio)},
      };
    }

  } else if (mParam.process_param.strategy == STRATEGY_TYPE_NORESIZE_TILE) {
    mResizeInfo = {
        {static_cast<int>(mInputShape[W_IDX]), static_cast<int>(mInputShape[H_IDX])},
        {static_cast<int>(mInputShape[W_IDX]), static_cast<int>(mInputShape[H_IDX])},
    };
  } else if (mParam.process_param.strategy == STRATEGY_TYPE_RESIZE_FULL) {
    mResizeInfo = {
        {static_cast<int>(mInputShape[W_IDX]), static_cast<int>(mInputShape[H_IDX])},
        {static_cast<int>(ocd_input_w), static_cast<int>(ocd_input_h)},
    };
  }
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
  mBufManager.initBuffer(ORIGIN_INPUT_BUF, mInputShape[C_IDX] * mInputShape[H_IDX] * mInputShape[W_IDX], true);  // 8UC3, origin size
  mBufManager.initBuffer("testwarp", 100 * 32 * 32 * sizeof(float), true);  // 8UC3, origin size
  if (mParam.ocr_param.type != nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP) {
    mBufManager.initBuffer(OCR_INPUT_BUF, mInputShape[W_IDX] * mInputShape[H_IDX] * sizeof(float), true);  // 32FC1, origin size
  } else {
    mBufManager.initBuffer(OCR_INPUT_BUF, mInputShape[W_IDX] * mInputShape[H_IDX] * 3 * sizeof(float), true); 
  }
}

void nvOCDR::preprocessInputImage() {

  mPreprocessTimer.Start();

  if (mResizeInfo.first == mResizeInfo.second) {
    LOG(INFO) << "no resize: " << mResizeInfo.first;
  } else {
    LOG(INFO) << "resize: " << mResizeInfo.first << " --> " << mResizeInfo.second;
  }

  // cv::resize(mInputImage, mInputImageResized, mResizeInfo.second);
  // mInputImageResized = mInputImage.clone();
  // LOG(ERROR) << mInputImageResized.size();

  // resize image to destinate size
  mOCDScoreMap = cv::Mat(mResizeInfo.second, CV_32F, cv::Scalar(0.F));
  mOCDValidCntMap = cv::Mat(mResizeInfo.second, CV_32F, cv::Scalar(0.F));

  // ocd input preprocess, bgr normalize
  // mInputImageResized.convertTo(mInputImageResized32F, CV_32FC3);
  // if (mParam.ocd_param.type == nvOCDParam::OCD_MODEL_TYPE::OCD_MODEL_TYPE_MIXNET) {
  //   mInputImageResized32F = mInputImageResized32F / 255.F;
  //   mInputImageResized32F -= cv::Scalar(IMG_MEAN_B_MIXNET, IMG_MEAN_G_MIXNET, IMG_MEAN_R_MIXNET);
  //   mInputImageResized32F = mInputImageResized32F.mul(cv::Scalar(IMG_MEAN_B_STD_MIXNET, IMG_MEAN_G_STD_MIXNET, IMG_MEAN_R_STD_MIXNET));
  // } else {
  //   mInputImageResized32F -= cv::Scalar(IMG_MEAN_B, IMG_MEAN_G, IMG_MEAN_R);
  //   mInputImageResized32F = mInputImageResized32F.mul(cv::Scalar(IMG_SCALE_BRG, IMG_SCALE_BRG, IMG_SCALE_BRG));
  // }

  launch_preprocess_gray(
    static_cast<uint8_t*>(mBufManager.getBuffer(ORIGIN_INPUT_BUF, DEVICE)),
    static_cast<float*>(mBufManager.getBuffer(OCR_INPUT_BUF, DEVICE)),
    mResizeInfo.first.width, mResizeInfo.first.height, 
    mResizeInfo.first.width, mResizeInfo.first.height, OCR_PREPROC_PARAM, true, mStream
  );
  mInputGrayImage = cv::Mat(mResizeInfo.first.height, mResizeInfo.first.width, CV_32F, 
    static_cast<float*>(mBufManager.getBuffer(OCR_INPUT_BUF, HOST))
  );
  mBufManager.copyDeviceToHost(OCR_INPUT_BUF, mStream);

  // cudaStreamSynchronize(mStream);
  // float* a = static_cast<float*>(mBufManager.getBuffer(OCR_INPUT_BUF, HOST));
  // cv::Mat gray(mResizeInfo.first.height, mResizeInfo.first.width, CV_32F, a);
  // cv::imwrite("gray.png", denormalizeGray(gray));


  // // ocr input preprocess, gray normalize
  // cv::cvtColor(mInputImage, mInputGrayImage, cv::COLOR_BGR2GRAY);
  // mInputGrayImage.convertTo(mInputGrayImage, CV_32F);
  // mInputGrayImage -= IMG_MEAN_GRAY;
  // mInputGrayImage = mInputGrayImage.mul(IMG_SCALE_GRAY);
  LOG(INFO) << "preprocess image time: " << mPreprocessTimer;
};

void nvOCDR::processTile(const nvOCDRInput& input) {
  const auto ocd_input_h = mOCDInputSize.height;
  const auto ocd_input_w = mOCDInputSize.width;

  // parameterize stride ??
  getTilePlan(ocd_input_w, ocd_input_h, mResizeInfo.second.width, mResizeInfo.second.height,
              static_cast<size_t>(ocd_input_h * 0.95));
  
  size_t num_tiles = mTiles.size();
  size_t num_ocd_bs = mOCDProcessor->getBatchSize();
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
    // preprocessOCDTile(start_idx, end_idx);  // batch inference
    preprocessOCDTileGPU(start_idx, end_idx);

    mOCDProcessor->infer(false, mStream);
    // mOCDNet->syncMemory(false, false, mStream);
    cudaStreamSynchronize(mStream);
    postprocessOCDTile(start_idx, end_idx);  // restore to image
  }
  cv::inRange(mOCDScoreMap, mParam.process_param.binarize_lower_threshold,
              mParam.process_param.binarize_upper_threshold, mOCDOutputMask);
  cv::resize(~mOCDOutputMask, mOCDOutputMask, mResizeInfo.first);
  LOG(INFO) << "ocd process time: " << mOCDTimer;

  // 2. select and filter the proper text area candidates
  mSelectProcessTimer.Start();
  selectOCRCandidates();
  LOG(INFO) << "select text process time: " << mSelectProcessTimer;

  // 2. ocr process
  size_t num_ocr_bs = mOCRProcessor->getBatchSize();
  size_t num_ocr_runs =
      mNumTexts % num_ocr_bs == 0 ? mNumTexts / num_ocr_bs : mNumTexts / num_ocr_bs + 1;
  LOG(INFO) << "found text area number: " << mNumTexts << ", run ocr " << num_ocr_runs
            << " times with batch size " << num_ocr_bs;

  // 3. ocr process 
  mOCRTimer.Start();
  for (size_t i = 0; i < num_ocr_runs; ++i) {
    size_t start_idx = i * num_ocr_bs;
    size_t end_idx = std::min((i + 1) * num_ocr_bs, mNumTexts);
    // recog on all specified directions
    for (auto const& bl_idx : mOCRDirections) {
      // mTmpTimer.Start();
      preprocessOCR(start_idx, end_idx, bl_idx);
      // LOG(INFO) << "preprocess ocr: " << mTmpTimer;
      mOCRProcessor->infer(true, mStream);
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
  cv::Size2f scale(static_cast<float>(mResizeInfo.first.width) / mResizeInfo.second.width,  static_cast<float>(mResizeInfo.first.height) / mResizeInfo.second.height);
  const auto output_h = mOCDInputSize.height;
  const auto output_w = mOCDInputSize.width;
  for (size_t j = start; j < end; j++) {
     const auto& tile = mTiles[j];
     if(mParam.ocd_param.type == nvOCDParam::OCD_MODEL_TYPE::OCD_MODEL_TYPE_NORMAL) {
       nvocdr::launch_preprocess_color(static_cast<uint8_t*>(mBufManager.getBuffer(ORIGIN_INPUT_BUF, DEVICE)),
                                 static_cast<float*>(mBufManager.getBuffer(mOCDProcessor->getInputBufName(), DEVICE)) + (j - start) * output_h * output_w * 3 , 
                                 mInputShape[W_IDX], mInputShape[H_IDX], tile, scale,
                                OCD_NORMAL_PREPROR_PARAM,
                                 true, true, 
                            mStream);
     } else if(mParam.ocd_param.type == nvOCDParam::OCD_MODEL_TYPE::OCD_MODEL_TYPE_MIXNET){
      nvocdr::launch_preprocess_color(static_cast<uint8_t*>(mBufManager.getBuffer(ORIGIN_INPUT_BUF, DEVICE)),
                          static_cast<float*>(mBufManager.getBuffer(mOCDProcessor->getInputBufName(), DEVICE)) + (j - start) * output_h * output_w * 3, 
                    mInputShape[W_IDX], mInputShape[H_IDX], tile, scale,
                    OCD_MIXNET_PREPROR_PARAM,
                    true, false, 
                    mStream);
     }
  }
}


void nvOCDR::preprocessOCDTile(size_t start, size_t end) {
  // float* buf = reinterpret_cast<float*>(
  //     mBufManager.getBuffer(mOCDNet->getBufName(OCDNET_INPUT), BUFFER_TYPE::HOST));

  // const auto ocd_input_h = mOCDNet->getInputH();
  // const auto ocd_input_w = mOCDNet->getInputW();
//   const auto ocd_input_h =0;
//   const auto ocd_input_w = 0;
//   // fill ocd batch, and normalize
//   for (size_t j = start; j < end; j++) {
//     const auto& tile = mTiles[j];
//     // 1 tile -> 1 batch in OCD
//     cv::Mat ocd_batch(cv::Size(ocd_input_w, ocd_input_h), CV_32FC3, cv::Scalar(0, 0, 0));
//     // LOG(INFO) << mInputImageResized32F.size() << "\t" << tile ;
//     mInputImageResized32F(tile).copyTo(ocd_batch(cv::Rect(cv::Point(0, 0), tile.br() - tile.tl())));

//     std::vector<cv::Mat> dummy_channels;
// #pragma unroll
//     for (size_t c = 0; c < 3; ++c) {
//       size_t offset = 0;
//       // we have input order bgr
//       // dcn use the bgr order, while mixnet use the rgb order
//       if (mParam.ocd_param.type == nvOCDParam::OCD_MODEL_TYPE::OCD_MODEL_TYPE_MIXNET) {
//         offset = (2 - c) * ocd_input_h * ocd_input_w;
//       } else {
//         offset = c * ocd_input_h * ocd_input_w;
//       }
//       dummy_channels.emplace_back(ocd_input_h, ocd_input_w, CV_32F, buf + offset, cv::Mat::AUTO_STEP);
//     }
//     cv::split(ocd_batch, dummy_channels);
//     buf += 3 * ocd_input_h * ocd_input_w;
//   }
}

void nvOCDR::selectOCRCandidates() {
  mNumTexts = 0;
  if (mParam.process_param.debug_image) {
    cv::imwrite("text_area.png", ~mOCDOutputMask);
  }
  mOCDProcessor->computeTextCandidates(mOCDOutputMask, &mQuadPts, &mTexts, &mNumTexts,
                                 mParam.process_param);
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

  float* buf = reinterpret_cast<float*>(
      mBufManager.getBuffer(mOCRProcessor->getInputBufName(), BUFFER_TYPE::HOST));
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
    cv::Mat text_roi(ocr_input_h, ocr_input_w, CV_32F, buf + ocr_input_h * ocr_input_w * (i - start));

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
    // mTmpTimer.Start();
    cv::warpPerspective(mInputGrayImage(rect), text_roi, h, cv::Size(ocr_input_w, ocr_input_h));
    // mTmpTimer.Stop();
    // // for debug
    if (mParam.process_param.debug_image) {
      // cv::Mat roi = mInputGrayImage(rect);
      cv::Mat warped(cv::Size(ocr_input_w, ocr_input_h), CV_32F);
      h = h.inv();

      std::vector<double> homo(9);
      for(size_t t = 0; t < 9 ; ++t) {
          homo[t] = h.at<double>(t / 3, t % 3);
      }

      launch_fused_preprocess_warp_perspective_gray(
        static_cast<uint8_t*>(mBufManager.getBuffer(ORIGIN_INPUT_BUF, DEVICE)),
        static_cast<float*>(mBufManager.getBuffer("testwarp", DEVICE)) + ocr_input_w * ocr_input_h * (i - start), 
        mResizeInfo.first.width, mResizeInfo.first.height,
        ocr_input_w, ocr_input_h, rect.tl().x, rect.tl().y, IMG_SCALE_GRAY, IMG_MEAN_GRAY, mStream,
        homo[0],homo[1],homo[2],homo[3],homo[4],homo[5],homo[6],homo[7],homo[8]

      );

      for(size_t t = 0; t < warped.rows; ++t) {
        for(size_t v = 0; v < warped.cols; ++v) {
          // cv::Vec3d src(v, t, 1);
          // cv::Mat dst = h * src;


          float dst_xf = homo[0] * v + homo[1] * t + homo[2];
          float dst_yf = homo[3] * v + homo[4] * t + homo[5];
          float c      = homo[6] * v + homo[7] * t + homo[8];

          int dst_x = (int)(dst_xf / c);
          int dst_y = (int)(dst_yf / c);

          warped.at<float>(t, v) = mInputGrayImage.at<float>(dst_y + rect.tl().y, dst_x + rect.tl().x);
          // LOG(INFO) << v << t << "  |  "<< dst.at<double>(0, 0) / dst.at<double>(2, 0) << " " << dst.at<double>(1, 0) / dst.at<double>(2, 0); 
          // break;
          // LOG(INFO) << t << v;
        }
        // break;
      }

      cv::imwrite("text_" + std::to_string(i) + "_" + std::to_string(bl_pt_idx) + ".png",
                  denormalizeGray(warped));
      // cv::imwrite("text_" + std::to_string(i) + "_" + std::to_string(bl_pt_idx) + "2.png",
      //             denormalizeGray(mInputGrayImage(rect)));
    }
    // break;
  }

  mBufManager.copyDeviceToHost("testwarp", mStream);
  float *buff = (float*)mBufManager.getBuffer("testwarp", HOST);

  for(size_t i = start; i < end; ++i) {
    cv::Mat aaa(ocr_input_h, ocr_input_w, CV_32F, buff + ocr_input_w * ocr_input_h * (i - start));

    cv::imwrite("warp_" + std::to_string(i) + "_" + std::to_string(bl_pt_idx) + ".png",
                  denormalizeGray(aaa));
  }
  
}

cv::Mat nvOCDR::denormalizeGray(const cv::Mat& input) {
  cv::Mat ret;
  input.copyTo(ret);
  ret /= IMG_SCALE_GRAY;
  ret += IMG_MEAN_GRAY;
  return ret;
}

void nvOCDR::postprocessOCR(size_t start, size_t end) {
  for (size_t i = start; i < end; ++i) {
    Text& text = mTexts[i];
    mOCRProcessor->decode(&text, i - start);
  }
}

nvOCDR::nvOCDR(const nvOCDRParam& param) : mParam(param) {
  // todo(shuohanc) check param
  if (param.input_shape[0] != 3 || param.input_shape[1] == 0 || param.input_shape[2] == 0) {
    throw std::runtime_error("input shape error");
  }

  cudaStreamCreate(&mStream);

  memcpy(mInputShape.data(), param.input_shape, mInputShape.size() * sizeof(size_t));


  LOG(INFO) << "input shape set to [" <<  mInputShape[0] << "," <<  mInputShape[1] << "," <<  mInputShape[2] <<"]";
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

  LOG(INFO) << "ocd model with input HxW: " << mOCDInputSize;
  LOG(INFO) << "ocr model with input HxW: " << mOCRInputSize;
  // warmp up
  for (size_t i = 0; i < NUM_WARMUP_RUNS; ++i) {
    mOCDProcessor->infer(true, mStream);
    mOCRProcessor->infer(true, mStream);
  }
  if (mParam.process_param.all_direction) {
    mOCRDirections = {0, 1, 2, 3};
  } else {
    mOCRDirections = {0};
  }
}

void nvOCDR::printTimeStat() {
  LOG(INFO) << "---------- time statistics ----------";
  LOG(INFO) << "history size: " << TIME_HISTORY_SIZE;

  auto preprocess_mean = mPreprocessTimer.getMean();
  auto ocd_mean = mOCDTimer.getMean();
  auto select_mean = mSelectProcessTimer.getMean();
  auto ocr_mean = mOCRTimer.getMean();
  auto e2e_mean = mE2ETimer.getMean();
  LOG(INFO) << "statistic shows in 'mean(ms) / percentage(%)' ";

  LOG(INFO) << "preproces: " << preprocess_mean << "ms / " << static_cast<int>(preprocess_mean / e2e_mean * 100) << "%"; 

  LOG(INFO) << "ocd: " << ocd_mean << "ms / " << static_cast<int>(ocd_mean / e2e_mean * 100) << "%"; 
  LOG(INFO) << "selector: " << select_mean << "ms / " << static_cast<int>(select_mean / e2e_mean * 100) << "%"; 
  LOG(INFO) << "ocr: " << ocr_mean << "ms / " << static_cast<int>(ocr_mean / e2e_mean * 100) << "%"; 

  LOG(INFO) << "e2e: " << e2e_mean <<"ms"; 
}

}  // namespace nvocdr

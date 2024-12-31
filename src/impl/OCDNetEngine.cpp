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

void OCDNetEngine::computeTextCandidatesMixNet(const cv::Mat& mask,
                                               std::vector<QUADANGLE>* const quads,
                                               std::vector<Text>* const texts, size_t* num_text,
                                               const ProcessParam& process_param) {
  Timer<100> timer;
  timer.Start();
  std::vector<std::vector<cv::Point>> raw_contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(~mask, raw_contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

//   cv::Mat viz(mask.size(), CV_8UC3, cv::Scalar(0, 0, 0));
//   cv::RNG rng(12345);

  LOG(ERROR) << "find contours: " << timer;
  std::vector<size_t> connected_compoment;
  std::vector<std::vector<size_t>> group;
  for (size_t i = 0; i < hierarchy.size(); ++i) {
    auto const& h = hierarchy[i];
    connected_compoment.push_back(i);
    if (h[0] == -1 && h[2] == -1) {
      group.push_back(connected_compoment);
      connected_compoment.clear();
      continue;
    }
    if (h[2] == -1 && h[3] == -1) {
      group.push_back(connected_compoment);
      connected_compoment.clear();
      continue;
    }
  }
  LOG(INFO) << "group: " << group.size();

  for (size_t i = 0; i < group.size(); ++i) {
    // cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    if (group[i].size() != 2) {
      continue;
    }
    auto outer = group[i][0];
    auto inner = group[i][1];

    if (hierarchy[outer][2] < 0) {
      std::swap(outer, inner);
    }

    // for(size_t j = 0;  j< group[i].size(); ++j) {
    //     cv::drawContours(viz, std::vector<std::vector<cv::Point>>{raw_contours[group[i][j]]}, -1, color);

    //     if (group[i][j] == inner) continue;
    //     cv::Vec4f line;
    //     cv::fitLine(raw_contours[group[i][j]], line,  cv::DIST_L2,0, 0.01,0.01);

    //     cv::Point p1(line[2] - 100 * line[0], line[3] - 100 * line[1]);

    //     // cv::Point p1(line[2] , line[3]);
    //     cv::Point p2(line[2] + 100 * line[0], line[3] + 100 * line[1]);

    //     cv::line(viz, p1, p2, color);

    // }
    auto rotated = cv::minAreaRect(raw_contours[outer]);

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

    if (*num_text == process_param.max_candidate) {
      break;
    }
  }
//   cv::imwrite("hctnr.png", viz);
}

// float OCDNetEngine::contourScore(const Mat& binary, const vector<Point>& contour)
// {
//     Rect rect = boundingRect(contour);
//     int xmin = max(rect.x, 0);
//     int xmax = min(rect.x + rect.width, binary.cols - 1);
//     int ymin = max(rect.y, 0);
//     int ymax = min(rect.y + rect.height, binary.rows - 1);

//     cv::Mat binROI = binary(Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));

//     cv::Mat mask = Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8U);
//     vector<Point> roiContour;
//     for (size_t i = 0; i < contour.size(); i++) {
//         Point pt = Point(contour[i].x - xmin, contour[i].y - ymin);
//         roiContour.push_back(pt);
//     }
//     vector<vector<Point>> roiContours = {roiContour};
//     fillPoly(mask, roiContours, Scalar(1));
//     float score = mean(binROI, mask).val[0];
//     return score;
// }

// void OCDNetEngine::unclip(const vector<Point2f>& inPoly, vector<Point2f> &outPoly)
// {
//     float area = contourArea(inPoly);

//     float length = arcLength(inPoly, true);
//     float distance = area * mUnclipRatio / length;

//     size_t numPoints = inPoly.size();
//     vector<vector<Point2f>> newLines;
//     for (size_t i = 0; i < numPoints; i++)
//     {
//         vector<Point2f> newLine;
//         Point pt1 = inPoly[i];
//         Point pt2 = inPoly[(i - 1) % numPoints];
//         Point vec = pt1 - pt2;
//         float unclipDis = (float)(distance / norm(vec));
//         Point2f rotateVec = Point2f(vec.y * unclipDis, -vec.x * unclipDis);
//         newLine.push_back(Point2f(pt1.x + rotateVec.x, pt1.y + rotateVec.y));
//         newLine.push_back(Point2f(pt2.x + rotateVec.x, pt2.y + rotateVec.y));
//         newLines.push_back(newLine);
//     }

//     size_t numLines = newLines.size();
//     for (size_t i = 0; i < numLines; i++)
//     {
//         Point2f a = newLines[i][0];
//         Point2f b = newLines[i][1];
//         Point2f c = newLines[(i + 1) % numLines][0];
//         Point2f d = newLines[(i + 1) % numLines][1];
//         Point2f pt;
//         Point2f v1 = b - a;
//         Point2f v2 = d - c;
//         float cosAngle = (v1.x * v2.x + v1.y * v2.y) / (norm(v1) * norm(v2));

//         if( fabs(cosAngle) > 0.7 )
//         {
//             pt.x = (b.x + c.x) * 0.5;
//             pt.y = (b.y + c.y) * 0.5;
//         }
//         else
//         {
//             float denom = a.x * (float)(d.y - c.y) + b.x * (float)(c.y - d.y) +
//                           d.x * (float)(b.y - a.y) + c.x * (float)(a.y - b.y);
//             float num = a.x * (float)(d.y - c.y) + c.x * (float)(a.y - d.y) + d.x * (float)(c.y - a.y);
//             float s = num / denom;

//             pt.x = a.x + s*(b.x - a.x);
//             pt.y = a.y + s*(b.y - a.y);
//         }
//         outPoly.push_back(pt);
//     }
// }

// OCDNetEngine::OCDNetEngine(const nvOCDParam& param)
//     // : mPolygonThreshold(polygonThreshold)
//     // , mUnclipRatio(unclipRatio)
//     // , mIsNHWC(isNHWC)
//     // , mBinaryThreshold(binaryThreshold)
//     // , mMaxContourNums(maxCandidates)
// {
//     // Init TRTEngine
//     mEngine = std::move(std::unique_ptr<TRTEngine>(new TRTEngine(param.engine_file)));
// }

// OCDNetEngine::~OCDNetEngine()
// {
//     mEngine.reset(nullptr);
// }

// bool
// OCDNetEngine::initTRTBuffer(BufferManager& buffer_mgr)
// {
//     // Init trt input gpu buffer
//     mTRTInputBufferIndex = buffer_mgr.initDeviceBuffer(mEngine->getMaxInputBufferSize(), sizeof(float));
//     mEngine->setInputBuffer(buffer_mgr.mDeviceBuffer[mTRTInputBufferIndex].data());

//     // Init trt output gpu buffer
//     mTRTOutputBufferIndex = buffer_mgr.initDeviceBuffer(mEngine->getMaxOutputBufferSize(), sizeof(float));
//     mEngine->setOutputBuffer(buffer_mgr.mDeviceBuffer[mTRTOutputBufferIndex].data());

//     mInferOutputbufHostIdx = buffer_mgr.initHostBuffer(mEngine->getMaxOutputBufferSize(), sizeof(float));
//     mOutputThresholdHostIdx = buffer_mgr.initHostBuffer(mEngine->getMaxOutputBufferSize(), sizeof(uchar));
//     mOutputThresholdDevIdx = buffer_mgr.initDeviceBuffer(mEngine->getMaxOutputBufferSize(), sizeof(uchar));
//     return 0;
// }

// bool
// OCDNetEngine::setInputShape(const Dims& input_shape)
// {
//     mEngine->setInputShape(input_shape);
//     return 0;
// }

// bool
// OCDNetEngine::setInputDeviceBuffer(DeviceBuffer& device_buffer, const int index)
// {
//     mTRTInputBufferIndex = index;
//     mEngine->setInputBuffer(device_buffer.data());
//     return 0;
// }

// bool
// OCDNetEngine::setOutputDeviceBuffer(DeviceBuffer& device_buffer, const int index)
// {
//     mTRTOutputBufferIndex = index;
//     mEngine->setOutputBuffer(device_buffer.data());
//     return 0;
// }

// // bool
// // OCDNetEngine::getIsNHWC()
// // {
// //     return mIsNHWC;
// // }

// // void
// // OCDNetEngine::setIsNHWC(bool order)
// // {
// //     mIsNHWC = order;
// // }

// bool
// OCDNetEngine::preprocess(void* input_data, const Dims& input_shape,
//                          void* output_data, const Dims& output_shape,
//                          const cudaStream_t& stream)
// {

//     blobFromImagesCUDA(input_data, output_data, input_shape, output_shape, mIsNHWC, stream);
//     return true;
// }

// bool
// OCDNetEngine::postprocess(BufferManager& buffer_mgr, const Dims& input_shape,
//                           std::vector<std::vector<Polygon>>& output,
//                           const cudaStream_t& stream)
// {

//     float* inferOutputDataHost =  static_cast<float*>(buffer_mgr.mHostBuffer[mInferOutputbufHostIdx].data());
//     uchar* thresholdCUDAHost = static_cast<uchar*>(buffer_mgr.mHostBuffer[mOutputThresholdHostIdx].data());

//     thresholdCUDA(buffer_mgr.mDeviceBuffer[mTRTOutputBufferIndex].data(), buffer_mgr.mDeviceBuffer[mOutputThresholdDevIdx].data(), input_shape, mBinaryThreshold, stream);
//     // copy thresholdCUDA results from device to host
//     checkCudaErrors(cudaMemcpyAsync(buffer_mgr.mHostBuffer[mOutputThresholdHostIdx].data(),  buffer_mgr.mDeviceBuffer[mOutputThresholdDevIdx].data(), volume(input_shape)*sizeof(uchar), cudaMemcpyDeviceToHost, stream));
//     checkCudaErrors(cudaMemcpyAsync(buffer_mgr.mHostBuffer[mInferOutputbufHostIdx].data(), buffer_mgr.mDeviceBuffer[mTRTOutputBufferIndex].data(), volume(input_shape) * sizeof(float), cudaMemcpyDeviceToHost, stream));
//     cudaStreamSynchronize(stream);

//     for (size_t n = 0; n < input_shape.d[0]; n++)
//     {
//         Mat binary = Mat(input_shape.d[2], input_shape.d[3], CV_32F, inferOutputDataHost+(input_shape.d[1])*(input_shape.d[2])*(input_shape.d[3])*n);
//         Mat bitmapCUDA = Mat(input_shape.d[2], input_shape.d[3], CV_8U, thresholdCUDAHost+(input_shape.d[1])*(input_shape.d[2])*(input_shape.d[3])*n);

//         // Find contours
//         findCoutourWarp(bitmapCUDA, binary, output, n);
//     }

//     return 0;

// }

// bool
// OCDNetEngine::infer(void* input_data, const Dims& input_shape,
//                     BufferManager& buffer_mgr, std::vector<std::vector<Polygon>>& output,
//                     const cudaStream_t& stream)
// {
//     preprocessWarp(input_data, input_shape, buffer_mgr, stream);

//     // Infer
//     mEngine->infer(stream);
//     postprocess(buffer_mgr, mEngine->getOutputShapeByName(OCDNET_OUTPUT),
//             output,stream);

//     return 0;
// }

// void
// OCDNetEngine::preprocessWarp(void* input_data, const Dims& input_shape,
//                     BufferManager& buffer_mgr,const cudaStream_t& stream)
// {
//      Dims infer_shape = mEngine->getExactInputShape();
//     if (mIsNHWC)
//     {
//         mScaleHeight = float(input_shape.d[1]) / float(infer_shape.d[2]);
//         mScaleWidth = float(input_shape.d[2]) / float(infer_shape.d[3]);
//     }
//     else
//     {
//         mScaleHeight = float(input_shape.d[2]) / float(infer_shape.d[2]);
//         mScaleWidth = float(input_shape.d[3]) / float(infer_shape.d[3]);
//     }

//     preprocess(input_data, input_shape,
//                buffer_mgr.mDeviceBuffer[mTRTInputBufferIndex].data(),
//                infer_shape, stream);
//     return;
// }

// void
// OCDNetEngine::preprocessAndThresholdWarpCUDA(void* input_data, const Dims& input_shape, BufferManager& buffer_mgr, Dims& ocdOutputPatchshape, const cudaStream_t& stream)
// {
//     preprocessWarp(input_data, input_shape, buffer_mgr, stream);
//     // Infer
//     mEngine->infer(stream);

//     ocdOutputPatchshape = mEngine->getOutputShapeByName(OCDNET_OUTPUT);
//     thresholdCUDA(buffer_mgr.mDeviceBuffer[mTRTOutputBufferIndex].data(), buffer_mgr.mDeviceBuffer[mOutputThresholdDevIdx].data(), ocdOutputPatchshape, mBinaryThreshold, stream);

//     return ;
// }

// void
// OCDNetEngine::findCoutourWarp(Mat& bitmapCUDA, Mat& binary, std::vector<std::vector<Polygon>>& output, const int outputIdx)
// {
//     Point2f vertex[4];
//     vector<Point>   contourScaled;
//     vector<Polygon> tempPoly;
//     vector<Point2f> approx;
//     vector<Point2f> polygon;
//     vector<vector<Point>> contours;
//     cv::findContours(bitmapCUDA, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

//     size_t numCandidate = min(contours.size(), (size_t)(mMaxContourNums > 0 ? mMaxContourNums : INT_MAX));
//     for (size_t i = 0; i < numCandidate; i++)
//     {
//         polygon.clear();
//         approx.clear();
//         contourScaled.clear();
//         auto contour = contours[i];
//         if (contourScore(binary, contour) < mPolygonThreshold)
//         {
//             continue;
//         }
//         // Rescale
//         for (size_t j = 0; j < contour.size(); j++)
//         {
//             contourScaled.emplace_back(Point(int(contour[j].x * mScaleWidth),
//                                             int(contour[j].y * mScaleHeight)));
//         }
//         // minAreaRect + Unclip
//         RotatedRect box = cv::minAreaRect(contourScaled);

//         // Filter the box with width or height < 1 pixel
//         float short_side = std::min(box.size.width, box.size.height);
//         if (short_side < 1)
//         {
//             continue;
//         }

//         // order: bottom left --> top left --> top right --> bottom right
//         box.points(vertex);
//         for (int j = 0; j < 4; j++)
//         {
//             approx.emplace_back(vertex[j]);
//         }

//         unclip(approx, polygon);
//         int x1 = polygon[0].x;
//         int y1 = polygon[0].y;
//         int x2 = polygon[1].x;
//         int y2 = polygon[1].y;
//         int x3 = polygon[2].x;
//         int y3 = polygon[2].y;
//         int x4 = polygon[3].x;
//         int y4 = polygon[3].y;
//         tempPoly.push_back({x1,y1,x2,y2,x3,y3,x4,y4});
//     }
//     output[outputIdx] = tempPoly;
//     return;
// }

}  // namespace nvocdr

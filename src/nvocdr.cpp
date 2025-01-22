#include <iostream>
#include <fstream>
#include <filesystem>

#include <nlohmann/json.hpp>
#include <glog/logging.h>


#include "impl/nvOCDR_impl.h"
#include "nvocdr.h"

void* nvOCDR_initialize(const nvOCDRParam& param) {
  // initial glog
  // google::InitGoogleLogging("nvOCDR");
  // FLAGS_logtostderr = 1;
  // FLAGS_colorlogtostderr = 1;

  nvocdr::nvOCDR* const handler = new nvocdr::nvOCDR(param);
  return reinterpret_cast<void*>(handler);
}

bool nvOCDR_process(void* const nvocdr_handler, const nvOCDRInput& input,
                    nvOCDROutput* const output) {
  nvocdr::nvOCDR* handler = reinterpret_cast<nvocdr::nvOCDR*>(nvocdr_handler);
  handler->process(input, output);
  return true;
}

void nvOCDR_print_stat(void* const nvocdr_handler) {
    nvocdr::nvOCDR* handler = reinterpret_cast<nvocdr::nvOCDR*>(nvocdr_handler);
    handler->printTimeStat();
}

inline nvOCDParam::OCD_MODEL_TYPE OCDStr2Enum(const std::string& type) {
  if (type == "mixnet") {
    return nvOCDParam::OCD_MODEL_TYPE::OCD_MODEL_TYPE_MIXNET;
  } else if(type == "normal") {
    return nvOCDParam::OCD_MODEL_TYPE::OCD_MODEL_TYPE_NORMAL;
  } else {
    throw std::runtime_error("ocd type not supported");
  }
}

inline std::string OCDEnum2Str(nvOCDParam::OCD_MODEL_TYPE type) {
  switch (type){
    case nvOCDParam::OCD_MODEL_TYPE::OCD_MODEL_TYPE_MIXNET: return "mixnet";
    case nvOCDParam::OCD_MODEL_TYPE::OCD_MODEL_TYPE_NORMAL: return "normal";
    default: throw std::runtime_error("ocd type not supported");
  }
}

inline std::string OCREnum2Str(nvOCRParam::OCR_MODEL_TYPE type) {
    switch (type){
      case nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_ATTN: return "ATTN";
      case nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CTC: return "CTC";
      case nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP: return "CLIP";
      default: throw std::runtime_error("ocr type not supported");
  }
}

inline nvOCRParam::OCR_MODEL_TYPE OCRStr2Enum(const std::string& type) {
  if (type == "ATTN") return nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_ATTN;
  if (type == "CTC") return nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CTC;
  if (type == "CLIP") return nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP;
  throw std::runtime_error("ocr type not supported " + type);
}

inline std::string StrategyEnum2Str(STRATEGY_TYPE strategy) {
    switch (strategy){
      case STRATEGY_TYPE::STRATEGY_TYPE_SMART: return "smart";
      case STRATEGY_TYPE::STRATEGY_TYPE_RESIZE_TILE: return "resize";
      case STRATEGY_TYPE::STRATEGY_TYPE_NORESIZE_TILE: return "no-resize";
      case STRATEGY_TYPE::STRATEGY_TYPE_RESIZE_FULL: return "full-resize";
      default: throw std::runtime_error("strategy not supported");
  }
}

inline STRATEGY_TYPE StrategyStr2Enum(const std::string& strategy) {
  if (strategy == "smart") return STRATEGY_TYPE::STRATEGY_TYPE_SMART;
  if (strategy == "resize") return STRATEGY_TYPE::STRATEGY_TYPE_RESIZE_TILE;
  if (strategy == "no-resize") return STRATEGY_TYPE::STRATEGY_TYPE_NORESIZE_TILE;
  if (strategy == "full-resize") return STRATEGY_TYPE::STRATEGY_TYPE_RESIZE_FULL;
  throw std::runtime_error("strategy not supported " + strategy);
}


void nvOCDR_dump_param(const nvOCDRParam& param, const char* param_path) {
  // nlohmann::json param;
  nlohmann::json ocd_param_j;
  nlohmann::json ocr_param_j;
  nlohmann::json process_param_j;
  nlohmann::json nvocdr_param_j;

  auto &ocd_param = param.ocd_param; 
  auto &ocr_param = param.ocr_param; 
  auto &process_param = param.process_param; 

  ocd_param_j["model_file"] = std::string(ocd_param.model_file);
  ocd_param_j["type"] = OCDEnum2Str(ocd_param.type);
  ocd_param_j["batch_size"] = ocd_param.batch_size;

  ocr_param_j["model_file"] = std::string(ocr_param.model_file);
  ocr_param_j["vocab_file"] = std::string(ocr_param.vocab_file);
  ocr_param_j["batch_size"] = ocr_param.batch_size;
  ocr_param_j["type"] = OCREnum2Str(ocr_param.type);

  process_param_j["strategy"] = StrategyEnum2Str(process_param.strategy);
  process_param_j["max_candidate"] = process_param.max_candidate;
  process_param_j["polygon_threshold"] = process_param.polygon_threshold;
  process_param_j["binarize_lower_threshold"] = process_param.binarize_lower_threshold;
  process_param_j["binarize_upper_threshold"] = process_param.binarize_upper_threshold;
  process_param_j["min_pixel_area"] = process_param.min_pixel_area;
  process_param_j["debug_log"] = process_param.debug_log;
  process_param_j["debug_image"] = process_param.debug_image;
  process_param_j["all_direction"] = process_param.all_direction;

  nvocdr_param_j["input_shape"] = {param.input_shape[0],param.input_shape[1],param.input_shape[2]};
  nvocdr_param_j["ocd_param"] = ocd_param_j;
  nvocdr_param_j["ocr_param"] = ocr_param_j;
  nvocdr_param_j["process_param"] = process_param_j;
  
  std::string dump_path(param_path);  
  std::ofstream out(param_path);
  out << std::setw(4) << nvocdr_param_j << std::endl;
}

void nvOCDR_parse_param(nvOCDRParam* param, const char* param_path) {
  std::ifstream in(param_path);
  nlohmann::json j;
  in >> j;
  LOG(INFO) << "parse param" << std::setw(4) << j;

  auto input_shape = j["input_shape"].get<std::vector<size_t>>();
  param->input_shape[0] = input_shape[0];
  param->input_shape[1] = input_shape[1];
  param->input_shape[2] = input_shape[2];

  auto &ocr_param_j = j["ocr_param"];
  param->ocr_param.batch_size = ocr_param_j["batch_size"].get<size_t>();
  param->ocr_param.type = OCRStr2Enum(ocr_param_j["type"].get<std::string>());
  strcpy(param->ocr_param.model_file, ocr_param_j["model_file"].get<std::string>().c_str());

  auto &ocd_param_j = j["ocd_param"];
  param->ocd_param.batch_size = ocd_param_j["batch_size"].get<size_t>();
  param->ocd_param.type = OCDStr2Enum(ocd_param_j["type"].get<std::string>());
  strcpy(param->ocd_param.model_file, ocd_param_j["model_file"].get<std::string>().c_str());
  
  auto &process_param_j = j["process_param"];
  param->process_param.strategy = StrategyStr2Enum(process_param_j["strategy"].get<std::string>());
  param->process_param.max_candidate = process_param_j["max_candidate"].get<size_t>();
  param->process_param.polygon_threshold = process_param_j["polygon_threshold"].get<float>();
  param->process_param.binarize_lower_threshold = process_param_j["binarize_lower_threshold"].get<float>();
  param->process_param.binarize_upper_threshold = process_param_j["binarize_upper_threshold"].get<float>();
  param->process_param.min_pixel_area = process_param_j["min_pixel_area"].get<float>();
  param->process_param.debug_log = process_param_j["debug_log"].get<bool>();
  param->process_param.debug_image = process_param_j["debug_image"].get<bool>();
  param->process_param.all_direction = process_param_j["all_direction"].get<bool>();
}

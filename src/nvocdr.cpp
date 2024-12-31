#include <iostream>

#include <glog/logging.h>

#include "impl/nvOCDR_impl.h"
#include "nvocdr.h"

void* nvOCDR_initialize(const nvOCDRParam& param) {
  // initial glog
  // google::InitGoogleLogging("nvOCDR");
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;

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

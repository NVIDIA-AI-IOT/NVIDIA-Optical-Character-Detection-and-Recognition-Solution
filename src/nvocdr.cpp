#include <iostream>

#include <glog/logging.h>

#include "nvocdr.h"
#include "impl/nvOCDR.h"




nvOCDRp nvOCDR_initialize(const nvOCDRParam& param)
{
    google::InitGoogleLogging("nvOCDR");
    FLAGS_logtostderr = 1;
    // todo(shuohanc) smart ptr
    nvocdr::nvOCDR* const handler = new nvocdr::nvOCDR(param);
    return reinterpret_cast<void *>(handler);
}

nvOCDRStat nvOCDR_process(void * const nvocdr_handler, const nvOCDRInput& input, nvOCDROutput* const output) {
    nvocdr::nvOCDR* handler = reinterpret_cast<nvocdr::nvOCDR*>(nvocdr_handler);
    handler->process(input, output);
    return SUCCESS;
}

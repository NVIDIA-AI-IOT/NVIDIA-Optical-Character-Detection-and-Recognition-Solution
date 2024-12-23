#include <iostream>

#include "nvocdr.h"
#include "impl/nvOCDR.h"



// nvOCDRp nvOCDR_init(nvOCDRParam param)
// {
//     nvocdr::nvOCDR* nv_ptr = new nvocdr::nvOCDR(param);

//     return static_cast<nvOCDRp>(nv_ptr); 
// }

// void nvOCDR_deinit(nvOCDRp nvocdr_ptr)
// {
//     nvocdr::nvOCDR* ptr = static_cast<nvocdr::nvOCDR*>(nvocdr_ptr);
//     delete ptr;
// }

// nvOCDRStat nvOCDR_inference(nvOCDRInput input, nvOCDROutputMeta* output, nvOCDRp nvocdr_ptr)
// {
//     nvocdr::nvOCDR* ptr = static_cast<nvocdr::nvOCDR*>(nvocdr_ptr);

//     nvinfer1::Dims infer_shape;
//     infer_shape.nbDims = 4;
//     infer_shape.d[0] = input.shape[0];
//     infer_shape.d[1] = input.shape[1];
//     infer_shape.d[2] = input.shape[2];
//     infer_shape.d[3] = input.shape[3];

//     int32_t batch_size = input.shape[0];

//     if (batch_size > MAX_BATCH_SIZE)
//     {
//         printf("The input data's batch size exceeds the MAX_BATCH_SIZE %d of nvOCDR.", input.shape[0]);
//         return FAIL;
//     }

//     std::vector<std::vector<std::pair<nvocdr::Polygon, std::pair<std::string, float>>>> nvocdr_output;

//     //Do inference
// #if DEBUG
//     Polygon debug_poly;
//     debug_poly.x1 = 100, debug_poly.y1 = 100;
//     debug_poly.x2 = 300, debug_poly.y2 = 100;
//     debug_poly.x3 = 300, debug_poly.y3 = 300;
//     debug_poly.x4 = 100, debug_poly.y4 = 300;
//     Polygon debug_poly_1;
//     debug_poly_1.x1 = 400, debug_poly_1.y1 = 400;
//     debug_poly_1.x2 = 600, debug_poly_1.y2 = 400;
//     debug_poly_1.x3 = 600, debug_poly_1.y3 = 600;
//     debug_poly_1.x4 = 400, debug_poly_1.y4 = 600;
//     nvocdr_output.push_back({std::make_pair(debug_poly, "NVOCDRDEBUG_0_0")});
//     nvocdr_output.push_back({std::make_pair(debug_poly, "中文NVOCDRDEBUG_1_0"),
//                              std::make_pair(debug_poly_1, "中文NVOCDRDEBUG_1_1")});
// #else
//     ptr->infer(input.mem_ptr, infer_shape, nvocdr_output);
// #endif
//     //Construct output
//     output->batch_size = batch_size;

//     //Compute the total polys:
//     int32_t total_cnt = 0;
//     for(auto img_polys: nvocdr_output)
//     {
//         total_cnt += img_polys.size();
//     }

//     if(total_cnt == 0)
//     {
//         for(int32_t i = 0; i < batch_size; ++i)
//         {
//             output->text_cnt[i] = 0;
//         }
//         output->text_ptr = nullptr;
//         return SUCCESS;
//     }

//     output->text_ptr = static_cast<nvOCDROutputBlob*>(malloc(sizeof(nvOCDROutputBlob) * total_cnt));
//     int32_t text_id = 0;
//     for(int32_t i = 0; i < batch_size; ++i)
//     {
//         output->text_cnt[i] = nvocdr_output[i].size();
//         for(int32_t j = 0; j < output->text_cnt[i]; ++j)
//         {
//             nvocdr::Polygon temp_poly = nvocdr_output[i][j].first;
//             std::string temp_str = nvocdr_output[i][j].second.first;
//             float temp_conf = nvocdr_output[i][j].second.second;
//             nvOCDROutputBlob* temp_blob = &output->text_ptr[text_id];
//             temp_blob->poly_cnt = 8;
//             temp_blob->polys[0] = temp_poly.x1;
//             temp_blob->polys[1] = temp_poly.y1;
//             temp_blob->polys[2] = temp_poly.x2;
//             temp_blob->polys[3] = temp_poly.y2;
//             temp_blob->polys[4] = temp_poly.x3;
//             temp_blob->polys[5] = temp_poly.y3;
//             temp_blob->polys[6] = temp_poly.x4;
//             temp_blob->polys[7] = temp_poly.y4;
//             temp_blob->ch_len = temp_str.size();
//             strncpy(temp_blob->ch, temp_str.c_str(), (size_t) MAX_CHARACTER_LEN - 1);
//             temp_blob->conf = temp_conf;
//             text_id +=1;
//         }
//     }
 
//     return SUCCESS;
// }

// nvOCDRStat nvOCDR_high_resolution_inference(nvOCDRInput input, nvOCDROutputMeta* output, nvOCDRp nvocdr_ptr,
//                                             float overlap_rate)
// {
//     nvocdr::nvOCDR* ptr = static_cast<nvocdr::nvOCDR*>(nvocdr_ptr);

//     nvinfer1::Dims infer_shape;
//     infer_shape.nbDims = 4;
//     infer_shape.d[0] = input.shape[0];
//     infer_shape.d[1] = input.shape[1];
//     infer_shape.d[2] = input.shape[2];
//     infer_shape.d[3] = input.shape[3];

//     int32_t batch_size = input.shape[0];

//     if (batch_size > MAX_BATCH_SIZE)
//     {
//         printf("The input data's batch size exceeds the MAX_BATCH_SIZE %d of nvOCDR.", input.shape[0]);
//         return FAIL;
//     }

//     std::vector<std::vector<std::pair<nvocdr::Polygon, std::pair<std::string, float>>>> nvocdr_output;

//     //Do inference
// #if DEBUG
//     Polygon debug_poly;
//     debug_poly.x1 = 100, debug_poly.y1 = 100;
//     debug_poly.x2 = 300, debug_poly.y2 = 100;
//     debug_poly.x3 = 300, debug_poly.y3 = 300;
//     debug_poly.x4 = 100, debug_poly.y4 = 300;
//     Polygon debug_poly_1;
//     debug_poly_1.x1 = 400, debug_poly_1.y1 = 400;
//     debug_poly_1.x2 = 600, debug_poly_1.y2 = 400;
//     debug_poly_1.x3 = 600, debug_poly_1.y3 = 600;
//     debug_poly_1.x4 = 400, debug_poly_1.y4 = 600;
//     nvocdr_output.push_back({std::make_pair(debug_poly, "NVOCDRDEBUG_0_0")});
//     nvocdr_output.push_back({std::make_pair(debug_poly, "中文NVOCDRDEBUG_1_0"),
//                              std::make_pair(debug_poly_1, "中文NVOCDRDEBUG_1_1")});
// #else
//     ptr->wrapInferPatch(input.mem_ptr, infer_shape, overlap_rate, nvocdr_output);
// #endif
//     //Construct output
//     output->batch_size = batch_size;

//     //Compute the total polys:
//     int32_t total_cnt = 0;
//     for(auto img_polys: nvocdr_output)
//     {
//         total_cnt += img_polys.size();
//     }

//     if(total_cnt == 0)
//     {
//         for(int32_t i = 0; i < batch_size; ++i)
//         {
//             output->text_cnt[i] = 0;
//         }
//         output->text_ptr = nullptr;
//         return SUCCESS;
//     }

//     output->text_ptr = static_cast<nvOCDROutputBlob*>(malloc(sizeof(nvOCDROutputBlob) * total_cnt));
//     int32_t text_id = 0;
//     for(int32_t i = 0; i < batch_size; ++i)
//     {
//         output->text_cnt[i] = nvocdr_output[i].size();
//         for(int32_t j = 0; j < output->text_cnt[i]; ++j)
//         {
//             nvocdr::Polygon temp_poly = nvocdr_output[i][j].first;
//             std::string temp_str = nvocdr_output[i][j].second.first;
//             float temp_conf = nvocdr_output[i][j].second.second;
//             nvOCDROutputBlob* temp_blob = &output->text_ptr[text_id];
//             temp_blob->poly_cnt = 8;
//             temp_blob->polys[0] = temp_poly.x1;
//             temp_blob->polys[1] = temp_poly.y1;
//             temp_blob->polys[2] = temp_poly.x2;
//             temp_blob->polys[3] = temp_poly.y2;
//             temp_blob->polys[4] = temp_poly.x3;
//             temp_blob->polys[5] = temp_poly.y3;
//             temp_blob->polys[6] = temp_poly.x4;
//             temp_blob->polys[7] = temp_poly.y4;
//             temp_blob->ch_len = temp_str.size();
//             strncpy(temp_blob->ch, temp_str.c_str(), (size_t) MAX_CHARACTER_LEN - 1);
//             temp_blob->conf = temp_conf;
//             text_id +=1;
//         }
//     }
 
//     return SUCCESS;
// }


nvOCDRp nvOCDR_initialize(const nvOCDRParam& param)
{
    // todo(shuohanc) smart ptr
    nvocdr::nvOCDR* const handler = new nvocdr::nvOCDR(param);
    return reinterpret_cast<void *>(handler);
}

nvOCDRStat nvOCDR_process(void * const nvocdr_handler, const nvOCDRInput& input, const nvOCDROutput* output) {
    nvocdr::nvOCDR* handler = reinterpret_cast<nvocdr::nvOCDR*>(nvocdr_handler);
    handler->process(input, output);
    return SUCCESS;
}

// nvOCDRStat nvOCDR_add_input(void * const nvocdr_handler, const nvOCDRInput& input) {
//     std::shared_ptr<nvocdr::nvOCDR> handler{reinterpret_cast<nvocdr::nvOCDR*>(nvocdr_handler)};
//     handler->addInput(input);
//     return SUCCESS;
// }


// nvOCDRStat nvOCDR_get_output(void * const nvocdr_handler,const nvOCDROutputMeta* output) {
//     std::shared_ptr<nvocdr::nvOCDR> handler{reinterpret_cast<nvocdr::nvOCDR*>(nvocdr_handler)};
//     handler->getOuput(output);
//     return SUCCESS;
// }
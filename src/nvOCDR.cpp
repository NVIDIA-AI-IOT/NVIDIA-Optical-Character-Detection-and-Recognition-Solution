# include "nvocdr.h"
# include "nvOCDR.h"
# include <algorithm>
# include <cstring>
# define DEBUG 0

using namespace nvocdr;

//============================== C-interface for nvOCDR==========================================//

nvOCDRp nvOCDR_init(nvOCDRParam param)
{
    nvocdr::nvOCDR* nv_ptr = new nvocdr::nvOCDR(param);

    return static_cast<nvOCDRp>(nv_ptr); 
}

void nvOCDR_deinit(nvOCDRp nvocdr_ptr)
{
    nvocdr::nvOCDR* ptr = static_cast<nvocdr::nvOCDR*>(nvocdr_ptr);
    delete ptr;
}

nvOCDRStat nvOCDR_inference(nvOCDRInput input, nvOCDROutputMeta* output, nvOCDRp nvocdr_ptr)
{
    nvocdr::nvOCDR* ptr = static_cast<nvocdr::nvOCDR*>(nvocdr_ptr);

    nvinfer1::Dims infer_shape;
    infer_shape.nbDims = 4;
    infer_shape.d[0] = input.shape[0];
    infer_shape.d[1] = input.shape[1];
    infer_shape.d[2] = input.shape[2];
    infer_shape.d[3] = input.shape[3];

    int32_t batch_size = input.shape[0];

    if (batch_size > MAX_BATCH_SIZE)
    {
        printf("The input data's batch size exceeds the MAX_BATCH_SIZE %d of nvOCDR.", input.shape[0]);
        return FAIL;
    }

    std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>> nvocdr_output;

    //Do inference
#if DEBUG
    Polygon debug_poly;
    debug_poly.x1 = 100, debug_poly.y1 = 100;
    debug_poly.x2 = 300, debug_poly.y2 = 100;
    debug_poly.x3 = 300, debug_poly.y3 = 300;
    debug_poly.x4 = 100, debug_poly.y4 = 300;
    Polygon debug_poly_1;
    debug_poly_1.x1 = 400, debug_poly_1.y1 = 400;
    debug_poly_1.x2 = 600, debug_poly_1.y2 = 400;
    debug_poly_1.x3 = 600, debug_poly_1.y3 = 600;
    debug_poly_1.x4 = 400, debug_poly_1.y4 = 600;
    nvocdr_output.push_back({std::make_pair(debug_poly, "NVOCDRDEBUG_0_0")});
    nvocdr_output.push_back({std::make_pair(debug_poly, "中文NVOCDRDEBUG_1_0"),
                             std::make_pair(debug_poly_1, "中文NVOCDRDEBUG_1_1")});
#else
    ptr->infer(input.mem_ptr, infer_shape, nvocdr_output);
#endif
    //Construct output
    output->batch_size = batch_size;

    //Compute the total polys:
    int32_t total_cnt = 0;
    for(auto img_polys: nvocdr_output)
    {
        total_cnt += img_polys.size();
    }

    if(total_cnt == 0)
    {
        for(int32_t i = 0; i < batch_size; ++i)
        {
            output->text_cnt[i] = 0;
        }
        output->text_ptr = nullptr;
        return SUCCESS;
    }

    output->text_ptr = static_cast<nvOCDROutputBlob*>(malloc(sizeof(nvOCDROutputBlob) * total_cnt));
    int32_t text_id = 0;
    for(int32_t i = 0; i < batch_size; ++i)
    {
        output->text_cnt[i] = nvocdr_output[i].size();
        for(int32_t j = 0; j < output->text_cnt[i]; ++j)
        {
            Polygon temp_poly = nvocdr_output[i][j].first;
            std::string temp_str = nvocdr_output[i][j].second.first;
            float temp_conf = nvocdr_output[i][j].second.second;
            nvOCDROutputBlob* temp_blob = &output->text_ptr[text_id];
            temp_blob->poly_cnt = 8;
            temp_blob->polys[0] = temp_poly.x1;
            temp_blob->polys[1] = temp_poly.y1;
            temp_blob->polys[2] = temp_poly.x2;
            temp_blob->polys[3] = temp_poly.y2;
            temp_blob->polys[4] = temp_poly.x3;
            temp_blob->polys[5] = temp_poly.y3;
            temp_blob->polys[6] = temp_poly.x4;
            temp_blob->polys[7] = temp_poly.y4;
            temp_blob->ch_len = temp_str.size();
            strncpy(temp_blob->ch, temp_str.c_str(), (size_t) MAX_CHARACTER_LEN - 1);
            temp_blob->conf = temp_conf;
            text_id +=1;
        }
    }
 
    return SUCCESS;
}

nvOCDRStat nvOCDR_high_resolution_inference(nvOCDRInput input, nvOCDROutputMeta* output, nvOCDRp nvocdr_ptr,
                                            float overlap_rate)
{
    nvocdr::nvOCDR* ptr = static_cast<nvocdr::nvOCDR*>(nvocdr_ptr);

    nvinfer1::Dims infer_shape;
    infer_shape.nbDims = 4;
    infer_shape.d[0] = input.shape[0];
    infer_shape.d[1] = input.shape[1];
    infer_shape.d[2] = input.shape[2];
    infer_shape.d[3] = input.shape[3];

    int32_t batch_size = input.shape[0];

    if (batch_size > MAX_BATCH_SIZE)
    {
        printf("The input data's batch size exceeds the MAX_BATCH_SIZE %d of nvOCDR.", input.shape[0]);
        return FAIL;
    }

    std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>> nvocdr_output;

    //Do inference
#if DEBUG
    Polygon debug_poly;
    debug_poly.x1 = 100, debug_poly.y1 = 100;
    debug_poly.x2 = 300, debug_poly.y2 = 100;
    debug_poly.x3 = 300, debug_poly.y3 = 300;
    debug_poly.x4 = 100, debug_poly.y4 = 300;
    Polygon debug_poly_1;
    debug_poly_1.x1 = 400, debug_poly_1.y1 = 400;
    debug_poly_1.x2 = 600, debug_poly_1.y2 = 400;
    debug_poly_1.x3 = 600, debug_poly_1.y3 = 600;
    debug_poly_1.x4 = 400, debug_poly_1.y4 = 600;
    nvocdr_output.push_back({std::make_pair(debug_poly, "NVOCDRDEBUG_0_0")});
    nvocdr_output.push_back({std::make_pair(debug_poly, "中文NVOCDRDEBUG_1_0"),
                             std::make_pair(debug_poly_1, "中文NVOCDRDEBUG_1_1")});
#else
    ptr->wrapInferPatch(input.mem_ptr, infer_shape, overlap_rate, nvocdr_output);
#endif
    //Construct output
    output->batch_size = batch_size;

    //Compute the total polys:
    int32_t total_cnt = 0;
    for(auto img_polys: nvocdr_output)
    {
        total_cnt += img_polys.size();
    }

    if(total_cnt == 0)
    {
        for(int32_t i = 0; i < batch_size; ++i)
        {
            output->text_cnt[i] = 0;
        }
        output->text_ptr = nullptr;
        return SUCCESS;
    }

    output->text_ptr = static_cast<nvOCDROutputBlob*>(malloc(sizeof(nvOCDROutputBlob) * total_cnt));
    int32_t text_id = 0;
    for(int32_t i = 0; i < batch_size; ++i)
    {
        output->text_cnt[i] = nvocdr_output[i].size();
        for(int32_t j = 0; j < output->text_cnt[i]; ++j)
        {
            Polygon temp_poly = nvocdr_output[i][j].first;
            std::string temp_str = nvocdr_output[i][j].second.first;
            float temp_conf = nvocdr_output[i][j].second.second;
            nvOCDROutputBlob* temp_blob = &output->text_ptr[text_id];
            temp_blob->poly_cnt = 8;
            temp_blob->polys[0] = temp_poly.x1;
            temp_blob->polys[1] = temp_poly.y1;
            temp_blob->polys[2] = temp_poly.x2;
            temp_blob->polys[3] = temp_poly.y2;
            temp_blob->polys[4] = temp_poly.x3;
            temp_blob->polys[5] = temp_poly.y3;
            temp_blob->polys[6] = temp_poly.x4;
            temp_blob->polys[7] = temp_poly.y4;
            temp_blob->ch_len = temp_str.size();
            strncpy(temp_blob->ch, temp_str.c_str(), (size_t) MAX_CHARACTER_LEN - 1);
            temp_blob->conf = temp_conf;
            text_id +=1;
        }
    }
 
    return SUCCESS;
}

//============================== Implementation of nvOCDR==========================================//

bool
nvOCDR::paramCheck()
{
    if (mParam.ocdnet_infer_input_shape[0] != 1 && mParam.ocdnet_infer_input_shape[0] != 3)
    {
        std::cerr<<"[ERROR] The OCDNet inference channel should be 1 or 3."<<std::endl;
        return false;
    }

    if (mParam.ocdnet_binarize_threshold <= 0 || mParam.ocdnet_binarize_threshold > 1)
    {
        std::cerr<<"[ERROR] The ocdnet_binarize_threshold should be (0, 1]."<<std::endl;
        return false;
    }
    
    if (mParam.ocdnet_polygon_threshold <= 0 || mParam.ocdnet_polygon_threshold > 1)
    {
        std::cerr<<"[ERROR] The ocdnet_polygon_threshold should be (0, 1]."<<std::endl;
        return false;
    }

    if (mParam.ocrnet_infer_input_shape[0] != 1 && mParam.ocrnet_infer_input_shape[0] != 3)
    {
        std::cerr<<"[ERROR] The OCRNet inference channel should be 1 or 3."<<std::endl;
        return false;
    }
    if (mParam.ocdnet_unclip_ratio <= 0)
    {
        std::cerr<<"[ERROR] The ocdnet_unclip_ratio should be large than 0"<<std::endl;
        return false;
    }


    return true;
}


nvOCDR::nvOCDR(nvOCDRParam param):
    mParam(param)
{
    if (!paramCheck())
    {
        std::cerr<<"[ERROR] The nvOCDR initialization failed due to wrong nvOCDRParam setting."<<std::endl;
        exit(0);
    }

    cudaStreamCreate(&mStream);

    mBuffMgr.mDeviceBuffer.clear();
    mBuffMgr.mHostBuffer.clear();

    if (param.input_data_format == NHWC)
        isNHWC = true;
    else
        isNHWC = false;

    //Init ocdnet
    std::string ocd_engine_path(param.ocdnet_trt_engine_path);
    mOCDNet = std::move(std::unique_ptr<OCDNetEngine>(new OCDNetEngine(ocd_engine_path,
                                                                       param.ocdnet_binarize_threshold,
                                                                       param.ocdnet_polygon_threshold,
                                                                       param.ocdnet_unclip_ratio,
                                                                       param.ocdnet_max_candidate,
                                                                       isNHWC)));
    // Init input and output buffer for OCDNet TRT inference:
    mOCDNet->initTRTBuffer(mBuffMgr);
    mOCDNetInputShape.nbDims=4;
    mOCDNetInputShape.d[0] = -1; // Dynamic batch size
    mOCDNetInputShape.d[1] = param.ocdnet_infer_input_shape[0];
    mOCDNetInputShape.d[2] = param.ocdnet_infer_input_shape[1];
    mOCDNetInputShape.d[3] = param.ocdnet_infer_input_shape[2];
    mOCDNetMaxBatch = mOCDNet->getMaxBatchSize();
    mOCDNet->setIsNHWC(isNHWC);

    bool upsidedown = param.upsidedown;

    DecodeMode decode_mode;
    if (param.ocrnet_decode == OCRNetDecode::CTC)
        decode_mode = DecodeMode::CTC;
    else if (param.ocrnet_decode == OCRNetDecode::Attention)
    {
        decode_mode = DecodeMode::Attention;
    }
    else if (param.ocrnet_decode == OCRNetDecode::CLIP)
    {
        decode_mode = DecodeMode::CLIP;
    }
    else
    {
        std::cerr << "[ERROR] Unsupported decode mode" << std::endl;
    }
        
    // Init ocrnet
    std::string ocr_engine_path(param.ocrnet_trt_engine_path);
    std::string ocr_dict_path(param.ocrnet_dict_file);
    mOCRNet = std::move(std::unique_ptr<OCRNetEngine>(new OCRNetEngine(ocr_engine_path,
                                                                       ocr_dict_path,
                                                                       upsidedown,
                                                                       decode_mode)));
    // Init input and output buffer for OCRNet TRT inference
    mOCRNet->initTRTBuffer(mBuffMgr);
    mOCRNetInputShape.nbDims=4;
    mOCRNetInputShape.d[0] = -1; // Dynamic batch size
    mOCRNetInputShape.d[1] = param.ocrnet_infer_input_shape[0];
    mOCRNetInputShape.d[2] = param.ocrnet_infer_input_shape[1];
    mOCRNetInputShape.d[3] = param.ocrnet_infer_input_shape[2];

    // Set the ocr max batch to 0.5 * real engine max batch if upsidedown = True
    if (upsidedown)
        mOCRNetMaxBatch = std::max(static_cast<int>(mOCRNet->getMaxBatchSize() * 0.5), 1);
    else
        mOCRNetMaxBatch = mOCRNet->getMaxBatchSize();

    int ocr_input_height = param.ocrnet_infer_input_shape[1];
    int ocr_input_width = param.ocrnet_infer_input_shape[2];
    mRect = std::move(std::unique_ptr<RectEngine>(new RectEngine(ocr_input_height, 
                                                                 ocr_input_width,
                                                                 mOCRNetMaxBatch,
                                                                 upsidedown, isNHWC,
                                                                 param.rotation_threshold,
                                                                 param.ocrnet_infer_input_shape[0])));

    // Set the OCRNet's TRT input buffer as rect's output
    mRect->setOutputBuffer(mOCRNet->mTRTInputBufferIndex);

    mRect->initBuffer(mBuffMgr);
}


nvOCDR::~nvOCDR()
{
    cudaStreamDestroy(mStream);
    mOCDNet.reset(nullptr);
    mRect.reset(nullptr);
    mOCRNet.reset(nullptr);
    mBuffMgr.mDeviceBuffer.clear();
    mBuffMgr.mHostBuffer.clear();
}


//@TODO(tylerz): 
// - Since the polygons are transfered through CPU,
//   the cudaStreamSynchronize(stream) is needed in the infer function.
// - Handle the input data batch size > engine max batch size
// - ????? How to handle upside-down ?????? => Solution 0: set the mOCRNetMaxBatch to (0.5 * real engine max batch size). 
void
nvOCDR::inferV1(void* input_data, const Dims& input_shape,
                std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>>& output)
{
    // Do OCDNet detection
    int32_t ocd_batch_size = input_shape.d[0];
    std::vector<std::vector<Polygon>> polys(ocd_batch_size);
    mOCDNetInputShape.d[0] = ocd_batch_size;
    mOCDNet->setInputShape(mOCDNetInputShape);
    mOCDNet->infer(input_data, input_shape, mBuffMgr, polys, mStream);
    OCRNetInferWarp(input_data, input_shape, polys, output);
}


void
nvOCDR::infer(void* input_data, const Dims& input_shape,
              std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>>& output)
{
    int input_batch_size =  input_shape.d[0];

    if (input_batch_size > mOCDNetMaxBatch)
    {
        std::cout<< "[WARNING] The input batch size exceed the max batch size of OCDNet engine. " 
                 << "nvOCDR will do sequentially inference with max batch size of OCDNet engine." << std::endl;
    }

    // Compute the inference time
    int max_infer_cnt = input_batch_size / mOCDNetMaxBatch;
    std::vector<int> infer_batch(max_infer_cnt, mOCDNetMaxBatch);
    int left = input_batch_size % mOCDNetMaxBatch;
    if (left != 0)
        infer_batch.emplace_back(left);

    std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>> temp_output;
    output.clear();
    Dims cur_input_shape = input_shape;
    uint8_t* cur_input_data = reinterpret_cast<uint8_t *>(input_data);
    for(auto batch : infer_batch)
    {
        cur_input_shape.d[0] = batch;
        inferV1(reinterpret_cast<void *>(cur_input_data), cur_input_shape, temp_output);
        output.reserve(output.size() + std::distance(temp_output.begin(), temp_output.end()));
        output.insert(output.end(), temp_output.begin(), temp_output.end());
        cur_input_data += volume(cur_input_shape);
    }
}


void
nvOCDR::inferPatch(void* oriImgData, void* patchImgs, const Dims& patchImgsShape, const Dims& oriImgshape, const float overlapRate,
              std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>>& output)
{
    int ori_w = 0;
    int ori_h = 0;
    int patch_w = 0;
    int patch_h = 0;
    Dims oriBinaryshape = oriImgshape;
    if(isNHWC)
    {
        ori_w = oriImgshape.d[2];
        ori_h = oriImgshape.d[1];
        patch_w = patchImgsShape.d[2];
        patch_h = patchImgsShape.d[1];
        oriBinaryshape.d[3] = 1;
    }
    else
    {
        ori_w = oriImgshape.d[3];
        ori_h = oriImgshape.d[2];
        patch_w = patchImgsShape.d[3];
        patch_h = patchImgsShape.d[2];
        oriBinaryshape.d[1] = 1;
    }

    int patchBatchSize =  patchImgsShape.d[0];
    int oriBatchSize =  oriImgshape.d[0];
    if (patchBatchSize > mOCDNetMaxBatch)
    {
        std::cout<< "[WARNING] The input batch size exceed the max batch size of OCDNet engine. " 
                 << "nvOCDR will do sequentially inference with max batch size of OCDNet engine." << std::endl;
    }

    // Compute the inference time
    int max_infer_cnt = patchBatchSize / mOCDNetMaxBatch;
    std::vector<int> infer_batch(max_infer_cnt, mOCDNetMaxBatch);
    int left = patchBatchSize % mOCDNetMaxBatch;
    if (left != 0)
        infer_batch.emplace_back(left);

    std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>> temp_output;
    output.clear();
    Dims cur_input_shape = patchImgsShape;
    uint8_t* cur_input_data = reinterpret_cast<uint8_t *>(patchImgs);

    // patch size are same with OCD input size. 
    int overlap_w = int(patch_w * overlapRate);
    int overlap_h = int(patch_h * overlapRate);

    int num_col_cut = int((ori_w - patch_w)/(patch_w - overlap_w)) + 1;
    int num_row_cut = int((ori_h - patch_h)/(patch_h - overlap_h)) + 1;
    cv::Mat oriImgBitMap(ori_h, ori_w, CV_8U); 
    cv::Mat oriImgMask(ori_h, ori_w, CV_32F); 
    DeviceBuffer oriImgBitMapDev( ori_w*ori_h, sizeof(uchar));
    DeviceBuffer oriImgMaskDev( ori_w*ori_h, sizeof(float));
    Dims ocdOutputPatchshape ;
    
    int ocdThresholdBufIdx = mOCDNet->getThresholdDevBufIdx();
    int ocdRawOutBufIdx = mOCDNet->getOcdRawOutDevBufIdx();
    int patchCnt = 0;
    for(auto batch : infer_batch)
    {

        cur_input_shape.d[0] = batch;
        mOCDNetInputShape.d[0] = batch;
        mOCDNet->setInputShape(mOCDNetInputShape);
        mOCDNet->preprocessAndThresholdWarpCUDA(cur_input_data, cur_input_shape, mBuffMgr, ocdOutputPatchshape, mStream);

        for(int i=0 ; i< batch; i++)
        {
            int row_idx = patchCnt / num_col_cut;
            int col_idx = patchCnt % num_col_cut;
            ocdOutputPatchshape.d[0] = 1;
            patchMaskMergeCUDA(oriImgBitMapDev.data(), mBuffMgr.mDeviceBuffer[ocdThresholdBufIdx].data()+(i*patch_w*patch_h), oriImgMaskDev.data(), mBuffMgr.mDeviceBuffer[ocdRawOutBufIdx].data()+(i*patch_w*patch_h*sizeof(float)), ocdOutputPatchshape, oriBinaryshape, overlapRate, col_idx, row_idx, num_col_cut, num_row_cut,mStream);
            patchCnt++;
        }
        cur_input_data += volume(cur_input_shape);
    }

    checkCudaErrors(cudaMemcpyAsync(oriImgBitMap.data,  oriImgBitMapDev.data(), oriImgBitMapDev.nbBytes(), cudaMemcpyDeviceToHost,mStream));
    checkCudaErrors(cudaMemcpyAsync(oriImgMask.data,  oriImgMaskDev.data(), oriImgMaskDev.nbBytes(), cudaMemcpyDeviceToHost,mStream));

#ifdef TRITON_DEBUG
    cudaStreamSynchronize(mStream);
    std::string pt_img_file = "./debug_img/0_merge_cuda.jpg";
    cv::imwrite(pt_img_file, oriImgBitMap); 
#endif

    std::vector<std::vector<Point>> contours;
    std::vector<std::vector<Polygon>> polys(oriBatchSize);
    mOCDNet->findCoutourWarp(oriImgBitMap, oriImgMask, polys, 0);
    OCRNetInferWarp(oriImgData, oriImgshape, polys, output);
}


void
nvOCDR::OCRNetInferWarp(void* input_data, const Dims& input_shape, std::vector<std::vector<Polygon>>& polys, std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>>& output)
{
    int ocd_batch_size = polys.size();
    // OCR Infer
    std::vector<int> valid_img_ids;
    std::vector<int> polys_to_img;
    std::vector<Polygon> flat_polys;
    int32_t ocr_batch_size = 0;

    for (auto i = 0; i < polys.size(); ++i)
    {
        if (polys[i].size() > 0)
        {
            valid_img_ids.emplace_back(i);
            ocr_batch_size += polys[i].size();

            flat_polys.reserve(flat_polys.size() + std::distance(polys[i].begin(), polys[i].end()));
            flat_polys.insert(flat_polys.end(), polys[i].begin(), polys[i].end());

            std::vector<int> cur_poly_to_img(polys[i].size(), i);
            polys_to_img.reserve(polys_to_img.size() + std::distance(cur_poly_to_img.begin(), cur_poly_to_img.end()));
            polys_to_img.insert(polys_to_img.end(), cur_poly_to_img.begin(), cur_poly_to_img.end());
        }
    }

    // Check if there is valid images
    if (valid_img_ids.size() == 0)
    {
        output.clear();
        output.resize(ocd_batch_size);
        return;
    }

    // Compute the inference time for rect + ocrnet
    int ocr_max_infer_cnt = ocr_batch_size / mOCRNetMaxBatch;
    std::vector<int> ocr_infer_batch(ocr_max_infer_cnt, mOCRNetMaxBatch);
    int left = ocr_batch_size % mOCRNetMaxBatch;
    if (left != 0)
        ocr_infer_batch.emplace_back(left);
    
    // Do multiple rect + ocrnet inference
    std::vector<std::pair<std::string, float>> de_texts;
    int32_t batch_start = 0;
    for(auto batch: ocr_infer_batch)
    {
        std::vector<Polygon> cur_polys(flat_polys.begin() + batch_start,
                                       flat_polys.begin() + batch_start + batch);
        
        std::vector<int> cur_polys_to_imgs(polys_to_img.begin() + batch_start,
                                           polys_to_img.begin() + batch_start + batch);
        batch_start += batch;
       
        // Do rectification
        if(!(mRect->infer(input_data, input_shape, mBuffMgr, cur_polys, cur_polys_to_imgs, mStream)))
        {
            std::cerr<< "[ERROR] Rectification falied. OCRNet inference will not launch" << std::endl;
            continue;
        }
        // Do OCR inference
        std::vector<std::pair<std::string, float>> cur_de_texts;
        int ocr_batch_size;
        if (mParam.upsidedown)
            ocr_batch_size = batch * 2;
        else
            ocr_batch_size = batch;
        mOCRNetInputShape.d[0] = ocr_batch_size;
        mOCRNet->setInputShape(mOCRNetInputShape);
        mOCRNet->infer(mBuffMgr, cur_de_texts, mStream);
        // Add 
        de_texts.reserve(de_texts.size() + std::distance(cur_de_texts.begin(), cur_de_texts.end()));
        de_texts.insert(de_texts.end(), cur_de_texts.begin(), cur_de_texts.end());
    }

    //Construct the output
    output.clear();
    output.resize(ocd_batch_size);
    int32_t text_id = 0;
    for(auto i : valid_img_ids) // batch-level
    {   
        std::vector<Polygon>& temp_poly = polys[i];
        for(auto j = 0; j < temp_poly.size(); ++j) // img-level
        {
            output[i].emplace_back(std::make_pair(temp_poly[j], de_texts[text_id]));
            text_id += 1;
        }
    }
}

void
nvOCDR::patchMaskMergeCUDA(void* oriThresholdData, void* patchThresholdData, void* oriRawData, void* patchRawData, const Dims& patchImgsShape, const Dims& oriImgshape, const float overlapRate, const int col_idx, const int row_idx, const int num_col_cut, const int num_row_cut ,const cudaStream_t& stream)
{
    int patch_w = patchImgsShape.d[3];
    int patch_h = patchImgsShape.d[2];
    int overlap_w = int(patch_w * overlapRate);
    int overlap_h = int(patch_h * overlapRate);
    ImgROI patchROI;
    ImgROI oriROI;

    if(row_idx == 0)
    {
        // top left patch
        if(col_idx == 0)
        {
            ImgROI tmpPatchRoi(0,0, int(patch_w-overlap_w/2), int(patch_h-overlap_h/2));
            ImgROI tmpOriRoi(0, 0, patch_w-overlap_w/2, patch_h-overlap_h/2);
            patchROI = tmpPatchRoi;
            oriROI = tmpOriRoi;
        }
        // top right patch
        else if(col_idx == num_col_cut-1)
        {

            ImgROI tmpPatchRoi(overlap_w/2, 0, patch_w-overlap_w/2, patch_h-overlap_h/2);
            ImgROI tmpOriRoi(col_idx*(patch_w-overlap_w) + overlap_w/2, 0, patch_w-overlap_w/2, patch_h-overlap_h/2);
            patchROI = tmpPatchRoi;
            oriROI = tmpOriRoi;
           
        }
        // top patches between left and right
        else 
        {
            ImgROI tmpPatchRoi(overlap_w/2, 0, patch_w-overlap_w, patch_h-overlap_h/2);
            ImgROI tmpOriRoi(col_idx*(patch_w-overlap_w)+overlap_w/2, 0,  patch_w-overlap_w, patch_h-overlap_h/2);
            patchROI = tmpPatchRoi;
            oriROI = tmpOriRoi;
        }   
    }
    else if (row_idx == num_row_cut - 1)
    {
        // bottom left patch
        if(col_idx == 0)
        {
            ImgROI tmpPatchRoi(0, overlap_h/2, patch_w-overlap_w/2, patch_h-overlap_h/2);
            ImgROI tmpOriRoi(0, row_idx*(patch_h-overlap_h)+overlap_h/2,  patch_w-overlap_w/2, patch_h-overlap_h/2);
            patchROI = tmpPatchRoi;
            oriROI = tmpOriRoi;
            
        }
        // bottom right patch
        else if(col_idx == num_col_cut-1)
        {
            ImgROI tmpPatchRoi(overlap_w/2, overlap_h/2, patch_w-overlap_w/2, patch_h-overlap_h/2);
            ImgROI tmpOriRoi(col_idx*(patch_w-overlap_w) + overlap_w/2, row_idx*(patch_h-overlap_h)+overlap_h/2, patch_w-overlap_w/2, patch_h-overlap_h/2);
            patchROI = tmpPatchRoi;
            oriROI = tmpOriRoi;
        }
        // bottom patches between left and right
        else 
        {
            ImgROI tmpPatchRoi(overlap_w/2, overlap_h/2, patch_w-overlap_w, patch_h-overlap_h/2);
            ImgROI tmpOriRoi(col_idx*(patch_w-overlap_w) + overlap_w/2, row_idx*(patch_h-overlap_h)+overlap_h/2,  patch_w-overlap_w, patch_h-overlap_h/2);
            patchROI = tmpPatchRoi;
            oriROI = tmpOriRoi;
        }   
    }
    else 
    {
        if (col_idx == 0)
        {
            ImgROI tmpPatchRoi(0,  overlap_h/2, patch_w-overlap_w/2, patch_h-overlap_h);
            ImgROI tmpOriRoi(0,  row_idx*(patch_h-overlap_h)+overlap_h/2, patch_w-overlap_w/2, patch_h-overlap_h);
            patchROI = tmpPatchRoi;
            oriROI = tmpOriRoi;
        }
        else if(col_idx == num_col_cut-1)
        {
            ImgROI tmpPatchRoi(overlap_w/2,overlap_h/2, patch_w-overlap_w/2, patch_h-overlap_h);
            ImgROI tmpOriRoi(col_idx*(patch_w-overlap_w) + overlap_w/2, row_idx*(patch_h-overlap_h)+overlap_h/2, patch_w-overlap_w/2, patch_h-overlap_h);
            patchROI = tmpPatchRoi;
            oriROI = tmpOriRoi;
        }
        else 
        {
            ImgROI tmpPatchRoi(overlap_w/2,overlap_h/2, patch_w-overlap_w, patch_h-overlap_h);
            ImgROI tmpOriRoi(col_idx*(patch_w-overlap_w) + overlap_w/2, row_idx*(patch_h-overlap_h)+overlap_h/2, patch_w-overlap_w, patch_h-overlap_h);
            patchROI = tmpPatchRoi;
            oriROI = tmpOriRoi;
        }   
    }

    patchMergeWarp( patchThresholdData, oriThresholdData, patchRawData, oriRawData, patchImgsShape, oriImgshape, patchROI, oriROI, stream );

    return;
}

void
nvOCDR::wrapInferPatch(void* input_data, const Dims& input_shape, float overlap_rate,
                       std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>>& output)
{
    //Compute the patch parameters
    int32_t patch_h = mParam.ocdnet_infer_input_shape[1];
    int32_t patch_w = mParam.ocdnet_infer_input_shape[1];
    int32_t batch_size = input_shape.d[0];
    if (batch_size!=1)
    {
        std::cerr<<"High resolution inference only supports bs=1"<<std::endl;
        exit(1);
    }
    int32_t orig_h = input_shape.d[1];
    int32_t orig_w = input_shape.d[2];
    int32_t orig_c = input_shape.d[3];
    int32_t overlap_h = overlap_rate * patch_h;
    int32_t overlap_w = overlap_rate * patch_w;
    int32_t crop_img_w = int32_t(ceilf(float(orig_w - patch_w)/float(patch_w - overlap_w)) * (patch_w - overlap_w) + patch_w);
    int32_t crop_img_h = int32_t(ceilf(float(orig_h - patch_h)/float(patch_h - overlap_h)) * (patch_h - overlap_h) + patch_h);
    int32_t crop_img_size = crop_img_h * crop_img_w * orig_c;
    Dims crop_input_shape;
    crop_input_shape.nbDims=4;
    crop_input_shape.d[0] = batch_size;
    crop_input_shape.d[1] = crop_img_h;
    crop_input_shape.d[2] = crop_img_w;
    crop_input_shape.d[3] = orig_c;

    void* crop_img_ptr;
    checkCudaErrors(cudaMalloc(&crop_img_ptr, crop_img_size * sizeof(uchar)));

    int32_t num_col = int32_t(float(crop_img_w - patch_w)/float(patch_w - overlap_w)) + 1;
    int32_t num_row = int32_t(float(crop_img_h - patch_h)/float(patch_h - overlap_h)) + 1;
    Dims patches_shape;
    patches_shape.nbDims=4;
    patches_shape.d[0] = num_col * num_row;
    patches_shape.d[1] = patch_h;
    patches_shape.d[2] = patch_w;
    patches_shape.d[3] = orig_c;

    int32_t patch_img_size = patch_w * patch_h * orig_c;
    void* patch_imgs_ptr;
    checkCudaErrors(cudaMalloc(&patch_imgs_ptr, patch_img_size * num_col * num_row * sizeof(uchar)));

    float rescale = 1.0f;
    rescale = KeepAspectRatioResize(input_data, crop_img_ptr, input_shape,
                                    crop_img_h, crop_img_w, mStream);

    // // dump the image
    // std::vector<uint8_t> raw_data(crop_img_size);
    // // std::vector<uint8_t> raw_data(orig_h * orig_w * orig_c);
    // cudaMemcpy(raw_data.data(), crop_img_ptr, crop_img_size, cudaMemcpyDeviceToHost);
    // // cudaMemcpy(raw_data.data(), input_data, orig_h * orig_w * orig_c, cudaMemcpyDeviceToHost);
    // cv::Mat frame_out(crop_img_h, crop_img_w, CV_8UC3, raw_data.data());
    // // cv::Mat frame_out(orig_h, orig_w, CV_8UC3, raw_data.data());
    // std::string img_path = "./debug_img/debug_pcb.png";
    // cv::imwrite(img_path, frame_out);
    
    //Create the cropped patches
    int32_t x_start = 0;
    int32_t y_start = 0;
    uchar* patch_ptr = reinterpret_cast<uchar*>(patch_imgs_ptr);
    for(int i = 0; i < num_row; i++)
        for(int j = 0; j < num_col; j++)
        {
            x_start = int32_t(j*(patch_w - overlap_w));
            y_start = int32_t(i*(patch_h - overlap_h));
            uchar* cur_crop_ptr = reinterpret_cast<uchar*>(crop_img_ptr) + y_start * crop_img_w * orig_c + x_start * orig_c;
            checkCudaErrors(cudaMemcpy2D(patch_ptr, orig_c * patch_w * sizeof(uchar),
                                         cur_crop_ptr, orig_c * crop_img_w * sizeof(uchar),
                                         patch_w * orig_c * sizeof(uchar), patch_h, cudaMemcpyDeviceToDevice));
            // //dump the patch
            // std::vector<uint8_t> raw_data(patch_img_size);
            // cudaMemcpy(raw_data.data(), patch_ptr, patch_img_size, cudaMemcpyDeviceToHost);
            // cv::Mat frame_out(patch_h, patch_w, CV_8UC3, raw_data.data());
            // std::string img_path = "./debug_img/debug_pcb_" + std::to_string(i) + std::to_string(j) + ".png";
            // cv::imwrite(img_path, frame_out);

            patch_ptr += patch_img_size;
        }

    //Do infer patch
    std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>> crop_output;
    inferPatch(crop_img_ptr, patch_imgs_ptr, patches_shape, crop_input_shape, overlap_rate, crop_output);
    cudaStreamSynchronize(mStream);

    //Postprocess
    output.clear();
    output.resize(batch_size);
    for(auto& output_pair: crop_output[0])
    {
        auto poly = output_pair.first;
        poly.x1 *= rescale;
        poly.x2 *= rescale;
        poly.x3 *= rescale;
        poly.x4 *= rescale;
        poly.y1 *= rescale;
        poly.y2 *= rescale;
        poly.y3 *= rescale;
        poly.y4 *= rescale;
        output[0].emplace_back(std::make_pair(poly, output_pair.second));
    }
    //Destroy the resources:
    checkCudaErrors(cudaFree(crop_img_ptr));
    checkCudaErrors(cudaFree(patch_imgs_ptr));
}
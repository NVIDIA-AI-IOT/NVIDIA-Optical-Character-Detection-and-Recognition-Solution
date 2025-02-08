#include <iostream>
#include <fstream>
#include <vector>
#include "TRTEngine.h"
#include "NvInferPlugin.h"
#include "NvInfer.h"


using namespace nvinfer1;
using namespace nvocdr;

size_t nvocdr::volume(const nvinfer1::Dims& dim)
{
    size_t data_size = 1;
    for(int i = 0; i < dim.nbDims; i++)
    {
        data_size *= dim.d[i];
    }
    return data_size;
}

class Logger : public ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;


TRTEngine::TRTEngine(const std::string engine_path)
{
    initLibNvInferPlugins(&logger, "");
    mRuntime = std::move(TRTUniquePtr<IRuntime>(createInferRuntime(logger)));
    // mRuntime = std::move(std::unique_ptr<IRuntime>(createInferRuntime(logger), InferDeleter()));
    //load binary engine:
    std::ifstream engineFile(engine_path, std::ios::binary);
    if (engineFile.good() == false)
    {
        std::cerr << "Error reading serialized TensorRT engine: " << engine_path << std::endl;
    }
    engineFile.seekg(0, std::ifstream::end);
    int64_t fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<uint8_t> engineBlob(fsize);
    engineFile.read(reinterpret_cast<char*>(engineBlob.data()), fsize);
    if (engineFile.good() == false)
    {
        std::cerr << "Error reading serialized TensorRT engine: " << engine_path << std::endl;
    }

    mEngine = std::move(TRTUniquePtr<ICudaEngine>(mRuntime->deserializeCudaEngine(engineBlob.data(), engineBlob.size())));

    
    //Get the input's name and max shape and output's name of the engine:
    mOutputNames.clear();
    for(int i = 0; i < mEngine->getNbIOTensors(); ++i)
    {
        std::string tensorName(mEngine->getIOTensorName(i));
        auto tensorMode = mEngine->getTensorIOMode(tensorName.c_str());
        if (tensorMode == TensorIOMode::kINPUT)
        {
            mInputName = tensorName;
            //For non-dynamic engine?
            // Dims shape = mEngine->getTensorShape(tensorName.c_str());
            //For dynamic engine, get the max shape:
            mMaxInputShape = mEngine->getProfileShape(tensorName.c_str(), 0, OptProfileSelector::kMAX);
            mMaxBatchSize = mMaxInputShape.d[0];
        }
        else if (tensorMode == TensorIOMode::kOUTPUT)
        {
            mOutputNames.emplace_back(tensorName);
        }
    }

    mContext = std::move(TRTUniquePtr<IExecutionContext>(mEngine->createExecutionContext()));

    //Compute the max output shape:
    mContext->setInputShape(mInputName.c_str(), mMaxInputShape);
    mMaxOutputShapes.clear();
    for(int i = 0; i < mOutputNames.size(); ++i)
    {
        mMaxOutputShapes.emplace_back(mContext->getTensorShape(mOutputNames[i].c_str()));
    }

}


size_t 
TRTEngine::getMaxInputBufferSize()
{
    size_t data_size = 1;
    for(int i = 0; i < mMaxInputShape.nbDims; ++i)
    {
        data_size *= mMaxInputShape.d[i];
    }
    return data_size;
}


void
TRTEngine::setInputBuffer(void* buffer)
{
    mContext->setTensorAddress(mInputName.c_str(), buffer);
}

void
TRTEngine::setInputBufferbyName(void* buffer, std::string& tensorName)
{
    mContext->setTensorAddress(tensorName.c_str(), buffer);
}


void
TRTEngine::setInputShape(const Dims shape)
{
    mContext->setInputShape(mInputName.c_str(), shape);
    mExactInputShape = shape;
    mExactOutputShapes.clear();
    for(int i = 0; i < mOutputNames.size(); ++i)
    {
        mExactOutputShapes.emplace(mOutputNames[i], mContext->getTensorShape(mOutputNames[i].c_str()));
    }
}


void
TRTEngine::setInputBatchSizebyName(const std::string& tensorName, const int batch_size)
{
    nvinfer1::Dims maxInputShape = mEngine->getProfileShape(tensorName.c_str(), 0, OptProfileSelector::kMAX);
    maxInputShape.d[0] = batch_size;
    mContext->setInputShape(tensorName.c_str(), maxInputShape);
    mExactOutputShapes.clear();
    for(int i = 0; i < mOutputNames.size(); ++i)
    {
        mExactOutputShapes.emplace(mOutputNames[i], mContext->getTensorShape(mOutputNames[i].c_str()));
    }
}


size_t 
TRTEngine::getMaxOutputBufferSize()
{
    size_t total_size = 0;
    for(int i = 0; i < mMaxOutputShapes.size(); ++i)
    {
        size_t data_size = 1;
        for(int j = 0; j < mMaxOutputShapes[i].nbDims; ++j)
        {
            data_size *= mMaxOutputShapes[i].d[j];
        }
        total_size += data_size;
    }

    return total_size;
}


size_t 
TRTEngine::getMaxTrtIoTensorSizeByName(std::string& tensorName)
{
    nvinfer1::Dims tensorDims = mContext->getTensorShape(tensorName.c_str());

    size_t data_size = 1;
    for(int i = 0; i < tensorDims.nbDims; ++i)
    {
        if (tensorDims.d[i] == -1)
        {
            data_size *= mMaxBatchSize;
        }
        else
        {
            data_size *= tensorDims.d[i];
        }
        
    }
    return data_size;
}


size_t 
TRTEngine::getTrtIoTensorDtypeSizeByName(std::string& tensorName)
{

    nvinfer1::DataType dataType = mEngine->getTensorDataType(tensorName.c_str());
    return sizeof(dataType);
}


void
TRTEngine::setOutputBuffer(void* buffer)
{
    // @TODO(tylerz): hard code to float for version 0
    float* temp_ptr = reinterpret_cast<float*>(buffer);
    size_t offset = 0;
    for(int i = 0; i < mOutputNames.size(); ++i)
    {
        temp_ptr += offset;
        mContext->setTensorAddress(mOutputNames[i].c_str(), temp_ptr);
        size_t data_size = 1;
        for(int j = 0; j < mMaxOutputShapes[i].nbDims; ++j)
        {
            data_size *= mMaxOutputShapes[i].d[j];
        }
        offset += data_size;
    }
}


void TRTEngine::setOutputBufferByName(void* buffer, std::string& tensorName)
{
    mContext->setTensorAddress(tensorName.c_str(), buffer);
}


const void*
TRTEngine::getOutputAddr(std::string output_name)
{
    return mContext->getTensorAddress(output_name.c_str());
}


Dims
TRTEngine::getExactOutputShape(std::string output_name)
{
    return mExactOutputShapes[output_name];
}


bool 
TRTEngine::infer(const cudaStream_t& stream)
{
    mContext->enqueueV3(stream);
    return 0;
}


TRTEngine::~TRTEngine()
{
    mContext.reset(nullptr);
    mEngine.reset(nullptr);
    mRuntime.reset(nullptr);
}

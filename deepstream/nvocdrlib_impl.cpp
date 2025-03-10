/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#define DEBUG 0
#if DEBUG
#include <opencv2/opencv.hpp>
#endif
#include <iostream>
#include <fstream>
#include <thread>
#include <string.h>
#include <queue>
#include <mutex>
#include <stdexcept>
#include <condition_variable>
#include <cuda.h>
#include <cuda_runtime.h>

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gst-nvquery.h"
#include "gstnvdsmeta.h"
#include "gst-nvevent.h"
#include "nvdscustomusermeta.h"
#include "nvdsdummyusermeta.h"

#include "nvdscustomlib_base.hpp"
#include "cudaEGL.h"

#include "nvocdr.h"

#define FORMAT_NV12 "NV12"
#define FORMAT_RGBA "RGBA"
#define FORMAT_I420 "I420"
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"


inline bool CHECK_(cudaError_t e, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
      std::cout << "CUDA runtime error " << cudaGetErrorString(e) << " at line " << iLine << " in file " << szFile << std::endl;
        exit (-1);
        return false;
    }
    return true;
}
#define ck(call) CHECK_(call, __LINE__, __FILE__)

/* This quark is required to identify NvDsMeta when iterating through
 * the buffer metadatas */
static GQuark _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);

/* Strcture used to share between the threads */
struct PacketInfo {
  GstBuffer *inbuf;
  guint frame_num;
};

class nvOCDRAlgorithm : public DSCustomLibraryBase
{
public:
  nvOCDRAlgorithm() {
    m_vectorProperty.clear();
    outputthread_stopped = false;
  }

  /* Set Init Parameters */
  virtual bool SetInitParams(DSCustom_CreateParams *params);

  /* Set Custom Properties  of the library */
  virtual bool SetProperty(Property &prop);

  /* Pass GST events to the library */
  virtual bool HandleEvent(GstEvent *event);

  virtual char *QueryProperties ();

  /* Process Incoming Buffer */
  virtual BufferResult ProcessBuffer(GstBuffer *inbuf);

  /* Retrun Compatible Caps */
  virtual GstCaps * GetCompatibleCaps (GstPadDirection direction,
        GstCaps* in_caps, GstCaps* othercaps);

  gboolean hw_caps;

  /* Deinit members */
  ~nvOCDRAlgorithm();

private:
  /* Output Processing Thread, push buffer to downstream  */
  void OutputThread(void);

public:
  guint source_id = 0;
  guint m_frameNum = 0;
  bool outputthread_stopped = false;

  /* Output Thread Pointer */
  std::thread *m_outputThread = NULL;

  /* Queue and Lock Management */
  std::queue<PacketInfo> m_processQ;
  std::mutex m_processLock;
  std::condition_variable m_processCV;

  /* Aysnc Stop Handling */
  gboolean m_stop = FALSE;

  /* Vector Containing Key:Value Pair of Custom Lib Properties */
  std::vector<Property> m_vectorProperty;

  void *m_scratchNvBufSurface = NULL;

  int m_max_batch = 0;
  int m_batch_width = 0;
  int m_batch_height = 0;
  int m_gpu_id = 0;
  NvBufSurface m_temp_surf;
  NvBufSurface *m_process_surf;
  cudaStream_t m_convertStream;
  NvBufSurfTransformParams m_transform_params;
  NvBufSurfTransformConfigParams m_transform_config_params;
  CUgraphicsResource m_cuda_resource;
  CUeglFrame m_egl_frame;
  //nvOCDR param
  std::string m_OCDNetEnginePath;
  std::vector<int> m_OCDNetInferShape;
  float m_OCDNetBinarizeThresh;
  float m_OCDNetPolyThresh;
  float m_OCDNetUnclipRatio = 1.5;
  int m_OCDNetMaxCandidate;
  bool m_RectUpsideDown = false;
  bool m_IsHighResolution = false;
  float m_OverlapRatio = 0.5;
  std::string m_OCRNetEnginePath;
  std::string m_OCRNetDictPath;
  std::vector<int> m_OCRNetInferShape;
  OCRNetDecode m_OCRNetDecode = CTC;
  void* m_interbuffer = nullptr;

  nvOCDRp m_nvOCDRLib;
};

extern "C" IDSCustomLibrary *CreateCustomAlgoCtx(DSCustom_CreateParams *params);
// Create Custom Algorithm / Library Context
extern "C" IDSCustomLibrary *CreateCustomAlgoCtx(DSCustom_CreateParams *params)
{
  return new nvOCDRAlgorithm();
}

// Set Init Parameters
bool nvOCDRAlgorithm::SetInitParams(DSCustom_CreateParams *params)
{
  DSCustomLibraryBase::SetInitParams(params);
  GstStructure *s1 = NULL;
  NvBufSurfTransform_Error err = NvBufSurfTransformError_Success;
  cudaError_t cudaReturn;

  // BufferPoolConfig pool_config = {0};
  // GstStructure *s1 = NULL;
  // GstCapsFeatures *feature;
  // GstStructure *config = NULL;

  s1 = gst_caps_get_structure(m_inCaps, 0);
  m_gpu_id = params->m_gpuId;

  NvBufSurfaceCreateParams create_params = { 0 };

  create_params.gpuId = m_gpu_id;
  create_params.width = m_batch_width;
  create_params.height = m_batch_height;
  create_params.size = 0;
  create_params.isContiguous = 1;
  create_params.colorFormat = NVBUF_COLOR_FORMAT_BGR;
  create_params.layout = NVBUF_LAYOUT_PITCH;
  create_params.memType = NVBUF_MEM_DEFAULT;

  gst_structure_get_int (s1, "batch-size", &m_max_batch);

  if (m_max_batch == 0) {
      // If this component is placed before mux, batch-size value is not set
      // In this case make batch_size = 1
      m_max_batch = 1;
  }

  if (NvBufSurfaceCreate (&m_process_surf, m_max_batch,
          &create_params) != 0) {
    GST_ERROR ("Error: Could not allocate internal buffer pool for nvOCDR");
    return false;
  }

  if(m_process_surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
    if (NvBufSurfaceMapEglImage (m_process_surf, 0) != 0) {
      GST_ERROR ("Error:Could not map EglImage from NvBufSurface for nvOCDR");
      return false;
    }

    if (cuGraphicsEGLRegisterImage (&m_cuda_resource,
        m_process_surf->surfaceList[0].mappedAddr.eglImage,
        CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE) != CUDA_SUCCESS) {
        GST_ELEMENT_ERROR (m_element, STREAM, FAILED,
            ("Failed to register EGLImage in cuda\n"), (NULL));
        return false;
    }
    if (cuGraphicsResourceGetMappedEglFrame (&m_egl_frame,
        m_cuda_resource, 0, 0) != CUDA_SUCCESS) {
        GST_ELEMENT_ERROR (m_element, STREAM, FAILED,
            ("Failed to get mapped EGL Frame\n"), (NULL));
        return false;
    }
  }

  m_transform_params.src_rect = new NvBufSurfTransformRect[m_max_batch];
  m_transform_params.dst_rect = new NvBufSurfTransformRect[m_max_batch];
  for(int idx = 0; idx < m_max_batch; idx++)
  {
    // the m_batch_width and m_batch_height is got in getCompatibleCabs
    // from src. the src and trans is the same 
    m_transform_params.src_rect[idx] =
        {0, 0, m_batch_width, m_batch_height};
    m_transform_params.dst_rect[idx] =
        {0, 0, m_batch_width, m_batch_height};
  }

  m_transform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
  m_transform_params.transform_flip = NvBufSurfTransform_None;
  // TODO(tylerz): Shall we enable option for the interpolation type ?
  m_transform_params.transform_filter = NvBufSurfTransformInter_Default;

  m_transform_config_params.compute_mode = NvBufSurfTransformCompute_GPU;

  cudaReturn =
    cudaStreamCreateWithFlags (&m_convertStream, cudaStreamNonBlocking);
  if (cudaReturn != cudaSuccess) {
    GST_ELEMENT_ERROR (m_element, RESOURCE, FAILED,
        ("Failed to create cuda stream"),
        ("cudaStreamCreateWithFlags failed with error %s",
            cudaGetErrorName (cudaReturn)));
    return FALSE;
  }

  m_transform_config_params.gpu_id = m_gpu_id;
  m_transform_config_params.cuda_stream = m_convertStream;

  m_temp_surf.surfaceList = new NvBufSurfaceParams[m_max_batch];
  m_temp_surf.batchSize = m_max_batch;
  m_temp_surf.gpuId = m_gpu_id;

  //Init nvOCDR lib
  nvOCDRParam param;
  param.input_data_format = NHWC;
  param.ocdnet_trt_engine_path = (char *)m_OCDNetEnginePath.c_str();
  param.ocdnet_infer_input_shape[0] = m_OCDNetInferShape[0];
  param.ocdnet_infer_input_shape[1] = m_OCDNetInferShape[1];
  param.ocdnet_infer_input_shape[2] = m_OCDNetInferShape[2];
  param.ocdnet_binarize_threshold = m_OCDNetBinarizeThresh;
  param.ocdnet_polygon_threshold = m_OCDNetPolyThresh;
  param.ocdnet_max_candidate = m_OCDNetMaxCandidate;
  param.ocdnet_unclip_ratio = m_OCDNetUnclipRatio;
  param.upsidedown = m_RectUpsideDown;
  param.ocrnet_trt_engine_path = (char *)m_OCRNetEnginePath.c_str();
  param.ocrnet_dict_file = (char *)m_OCRNetDictPath.c_str();
  param.ocrnet_decode = m_OCRNetDecode;
  param.ocrnet_infer_input_shape[0] = m_OCRNetInferShape[0];
  param.ocrnet_infer_input_shape[1] = m_OCRNetInferShape[1];
  param.ocrnet_infer_input_shape[2] = m_OCRNetInferShape[2];

  // Init intermediate buffer if the nvbufsurface adds padding
  // interbuffer save BGR data
  uint32_t surf_width = m_process_surf->surfaceList->width;
  uint32_t surf_height = m_process_surf->surfaceList->height;
  uint32_t surf_size = m_process_surf->surfaceList->dataSize;
  uint32_t surf_channel = 3; //BGR
  if (surf_width*surf_height*surf_channel != surf_size)
  {
    ck(cudaMalloc(&m_interbuffer,
                  sizeof(uint8_t) * m_batch_width * m_batch_height * 3 * m_max_batch));
  }


#if DEBUG
  std::cout<<m_OCDNetEnginePath.c_str()<<std::endl;
  std::cout<<m_OCDNetInferShape[0]<<" "<<m_OCDNetInferShape[1]<<" "<<m_OCDNetInferShape[2]<<std::endl;
  std::cout<<m_OCDNetBinarizeThresh<<std::endl;
  std::cout<<m_OCDNetPolyThresh<<std::endl;
  std::cout<<m_OCDNetMaxCandidate<<std::endl;
  std::cout<<m_RectUpsideDown<<std::endl;
  std::cout<<m_OCRNetEnginePath.c_str()<<std::endl;
  std::cout<<m_OCRNetDictPath.c_str()<<std::endl;
  std::cout<<m_OCRNetInferShape[0]<<" "<<m_OCRNetInferShape[1]<<" "<<m_OCRNetInferShape[2]<<std::endl;
  std::cout<<"Max Batch Size: "<< m_max_batch << std::endl;
#endif

  m_nvOCDRLib = nvOCDR_init(param);

  m_outputThread = new std::thread(&nvOCDRAlgorithm::OutputThread, this);

  return true;
}

// Return Compatible Output Caps based on input caps
GstCaps* nvOCDRAlgorithm::GetCompatibleCaps (GstPadDirection direction,
        GstCaps* in_caps, GstCaps* othercaps)
{
  GstCaps* result = NULL;
  GstStructure *s1, *s2;
  //gint width, height;
  gint i, num, denom;
  const gchar *inputFmt = NULL;

  GST_DEBUG ("\n----------\ndirection = %d (1=Src, 2=Sink) -> %s:\n"
      "CAPS = %s\n", direction, __func__, gst_caps_to_string(in_caps));
  GST_DEBUG ("%s : OTHERCAPS = %s\n", __func__, gst_caps_to_string(othercaps));

  othercaps = gst_caps_truncate(othercaps);
  othercaps = gst_caps_make_writable(othercaps);

  //num_input_caps = gst_caps_get_size (in_caps);
  int num_output_caps = gst_caps_get_size (othercaps);

  // TODO: Currently it only takes first caps
  s1 = gst_caps_get_structure(in_caps, 0);
  for (i=0; i<num_output_caps; i++)
  {
    s2 = gst_caps_get_structure(othercaps, i);
    inputFmt = gst_structure_get_string (s1, "format");

    GST_DEBUG ("InputFMT = %s \n\n", inputFmt);

    // Check for desired color format
    if ((strncmp(inputFmt, FORMAT_NV12, strlen(FORMAT_NV12)) == 0) ||
            (strncmp(inputFmt, FORMAT_RGBA, strlen(FORMAT_RGBA)) == 0))
    {
      //Set these output caps
      gst_structure_get_int (s1, "width", &m_batch_width);
      gst_structure_get_int (s1, "height", &m_batch_height);

      /* otherwise the dimension of the output heatmap needs to be fixated */

      // Here change the width and height on output caps based on the
      // information provided by the custom library
      gst_structure_fixate_field_nearest_int(s2, "width", m_batch_width);
      gst_structure_fixate_field_nearest_int(s2, "height", m_batch_height);
      if (gst_structure_get_fraction(s1, "framerate", &num, &denom))
      {
        gst_structure_fixate_field_nearest_fraction(s2, "framerate", num,
            denom);
      }

      // TODO: Get width, height, coloutformat, and framerate from
      // customlibrary API set the new properties accordingly
      gst_structure_set (s2, "width", G_TYPE_INT, (gint)(m_batch_width), NULL);
      gst_structure_set (s2, "height", G_TYPE_INT, (gint)(m_batch_height),
          NULL);
      gst_structure_set (s2, "format", G_TYPE_STRING, inputFmt, NULL);

      result = gst_caps_ref(othercaps);
      gst_caps_unref(othercaps);
      GST_DEBUG ("%s : Updated OTHERCAPS = %s \n\n", __func__,
          gst_caps_to_string(othercaps));

      break;
    } else {
      continue;
    }
  }
  return result;
}

void split_str(
    const char* s,
    std::vector<std::string>& ret,  // NOLINT(runtime/references)
    char del = ',') {
    int idx = 0;
    auto p = std::string(s + idx).find(std::string(1, del));
    while (std::string::npos != p) {
        auto s_tmp = std::string(s + idx).substr(0, p);
        ret.push_back(s_tmp);
        idx += (p + 1);
        p = std::string(s + idx).find(std::string(1, del));
    }
    if (s[idx] != 0) {
        ret.push_back(std::string(s + idx));
    }
}

bool nvOCDRAlgorithm::HandleEvent (GstEvent *event)
{
  switch (GST_EVENT_TYPE(event))
  {
       case GST_EVENT_EOS:
           m_processLock.lock();
           m_stop = TRUE;
           m_processCV.notify_all();
           m_processLock.unlock();
           while (outputthread_stopped == FALSE)
           {
               //g_print ("waiting for processq to be empty, buffers in processq = %ld\n", m_processQ.size());
               g_usleep (1000);
           }
           break;
       default:
           break;
  }
  if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_STREAM_EOS)
  {
      gst_nvevent_parse_stream_eos (event, &source_id);
  }
  return true;
}


char *nvOCDRAlgorithm::QueryProperties ()
{
    char *str = new char[1000];
    strcpy (str, "nvOCDR LIBRARY PROPERTIES\n \t\t\tcustomlib-props=\"ocdnet-engine-path:x\""
            "\n\t\t\tcustomlib-props=\"ocdnet-input-shape:x,y,z\""
            "\n\t\t\tcustomlib-props=\"ocdnet-binarize-threshold:x\""
            "\n\t\t\tcustomlib-props=\"ocdnet-polygon-threshold:x\""
            "\n\t\t\tcustomlib-props=\"ocdnet-max-candidate:x\""
            "\n\t\t\tcustomlib-props=\"rectifier-upsidedown:x\""
            "\n\t\t\tcustomlib-props=\"is-high-resolution:x\""
            "\n\t\t\tcustomlib-props=\"overlap-ratio:x\""
            "\n\t\t\tcustomlib-props=\"ocrnet-engine-path:x\""
            "\n\t\t\tcustomlib-props=\"ocrnet-dict-path:x\""
            "\n\t\t\tcustomlib-props=\"ocrnet-decode:x\""
            "\n\t\t\tcustomlib-props=\"ocrnet-input-shape:x,y,z\"");
    return str;
}


// Set Custom Library Specific Properties
bool nvOCDRAlgorithm::SetProperty(Property &prop)
{
  std::cout << "Inside Custom Lib : Setting Prop Key=" << prop.key << " Value=" << prop.value << std::endl;
  m_vectorProperty.emplace_back(prop.key, prop.value);


  try
  {
      if (prop.key.compare("ocdnet-engine-path") == 0)
      {
          m_OCDNetEnginePath.assign(prop.value);
      }
      if (prop.key.compare("ocdnet-input-shape") == 0)
      {
          std::vector<std::string> temp_shape;
          split_str(prop.value.c_str(), temp_shape);
          m_OCDNetInferShape.clear();
          for(auto s: temp_shape)
          {
            m_OCDNetInferShape.emplace_back(stoi(s));
          }
      }
      if (prop.key.compare("ocdnet-binarize-threshold") == 0)
      {
          m_OCDNetBinarizeThresh = stof(prop.value);
      }
      if (prop.key.compare("ocdnet-polygon-threshold") == 0)
      {
          m_OCDNetPolyThresh = stof(prop.value);
      }
      if (prop.key.compare("ocdnet-unclip-ratio") == 0)
      {
          m_OCDNetUnclipRatio = stof(prop.value);
      }
      if (prop.key.compare("ocdnet-max-candidate") == 0)
      {
          m_OCDNetMaxCandidate = stoi(prop.value);
      }
      if (prop.key.compare("rectifier-upsidedown") == 0)
      {
          m_RectUpsideDown = stoi(prop.value);
      }
      if (prop.key.compare("is-high-resolution") == 0)
      {
          m_IsHighResolution = stoi(prop.value);
      }
      if (prop.key.compare("overlap-ratio") == 0)
      {
          m_OverlapRatio = stof(prop.value);
          if (m_OverlapRatio > 1 or m_OverlapRatio <=0)
          {
            printf("overlap-ratio should be in (0, 1]");
            exit(1);
          }
      }
      if (prop.key.compare("ocrnet-engine-path") == 0)
      {
          m_OCRNetEnginePath.assign(prop.value);
      }
      if (prop.key.compare("ocrnet-dict-path") == 0)
      {
          m_OCRNetDictPath.assign(prop.value);
      }
      if (prop.key.compare("ocrnet-decode") == 0)
      {
          std::string decode_mode;
          decode_mode.assign(prop.value);
          if (decode_mode == "CTC")
          {
            m_OCRNetDecode = CTC;
          }
          else if (decode_mode == "Attention")
          {
            m_OCRNetDecode = Attention;
          }
          else
          {
            printf("ocrnet-decode should be in [CTC, Attention]");
            exit(1);
          }
      }
      if (prop.key.compare("ocrnet-input-shape") == 0)
      {
          std::vector<std::string> temp_shape;
          split_str(prop.value.c_str(), temp_shape);
          m_OCRNetInferShape.clear();
          for(auto s: temp_shape)
          {
            m_OCRNetInferShape.emplace_back(stoi(s));
          }
      }
  }
  catch(std::invalid_argument& e)
  {
      std::cout << "Invalid Argument" << std::endl;
      return false;
  }
  return true;
}

/* Deinitialize the Custom Lib context */
nvOCDRAlgorithm::~nvOCDRAlgorithm()
{
    std::unique_lock<std::mutex> lk(m_processLock);
  //std::cout << "Process Q Empty : " << m_processQ.empty() << std::endl;
  m_processCV.wait(lk, [&]{return m_processQ.empty();});
  m_stop = TRUE;
  m_processCV.notify_all();
  lk.unlock();

  /* Wait for OutputThread to complete */
  if (m_outputThread) {
    m_outputThread->join();
  }

  if (m_process_surf)
      NvBufSurfaceDestroy (m_process_surf);

  delete[] m_transform_params.src_rect;
  delete[] m_transform_params.dst_rect;
  delete[] m_temp_surf.surfaceList;

  nvOCDR_deinit(m_nvOCDRLib);

  if (m_interbuffer != nullptr)
  {
    ck(cudaFree(m_interbuffer));
  }

  if (m_convertStream)
    cudaStreamDestroy (m_convertStream);

}


/* Process Buffer */
BufferResult nvOCDRAlgorithm::ProcessBuffer (GstBuffer *inbuf)
{
  GST_DEBUG("nvOCDRInfer: ---> Inside %s frame_num = %d\n", __func__,
      m_frameNum++);

  // Push buffer to process thread for further processing
  PacketInfo packetInfo;
  packetInfo.inbuf = inbuf;
  packetInfo.frame_num = m_frameNum;


  m_processLock.lock();
  m_processQ.push(packetInfo);
  m_processCV.notify_all();
  m_processLock.unlock();

  return BufferResult::Buffer_Async;
}

// TODO(tylerz): Add OCDR meta like object_detection ???
extern "C"
gboolean nvds_add_ocdr_meta(NvDsBatchMeta *batch_meta, nvOCDROutputMeta &output)
{
  nvds_acquire_meta_lock (batch_meta);

  // Iterate over the output
  int offset=0;
  NvDsMetaList * l_frame = NULL;
  NvDsObjectMeta *obj_meta = NULL;
  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) 
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    int batch_id = frame_meta->batch_id;
    for(int i = 0; i < output.text_cnt[batch_id]; ++i)
    {
      obj_meta = nvds_acquire_obj_meta_from_pool (batch_meta);
      nvOCDROutputBlob output_blob = output.text_ptr[offset];
      std::string text(output_blob.ch);
      int box_xmin = std::min(std::min(output_blob.polys[0], output_blob.polys[2]), 
                              std::min(output_blob.polys[4], output_blob.polys[6]));
      int box_ymin = std::min(std::min(output_blob.polys[1], output_blob.polys[3]),
                              std::min(output_blob.polys[5], output_blob.polys[7]));
      int box_xmax = std::max(std::max(output_blob.polys[0], output_blob.polys[2]), 
                              std::max(output_blob.polys[4], output_blob.polys[6]));
      int box_ymax = std::max(std::max(output_blob.polys[1], output_blob.polys[3]),
                              std::max(output_blob.polys[5], output_blob.polys[7]));
      int box_width = box_xmax - box_xmin;
      int box_height = box_ymax - box_ymin;
      // @TODO(tylerz): add constant meta.
      obj_meta->unique_component_id = 0;
      obj_meta->confidence = output_blob.conf;
      obj_meta->object_id = UNTRACKED_OBJECT_ID;
      obj_meta->class_id = 0;
      NvOSD_RectParams & rect_params = obj_meta->rect_params;
      NvOSD_TextParams & text_params = obj_meta->text_params;

      // @TODO(tylerz): Currently have to work as in primary detector position.
      rect_params.left = box_xmin;
      rect_params.top = box_ymin;
      rect_params.width = box_width;
      rect_params.height = box_height;
      rect_params.border_width = 3;
      rect_params.has_bg_color = 0;
      rect_params.border_color = (NvOSD_ColorParams) {1, 0, 0, 1};

      obj_meta->detector_bbox_info.org_bbox_coords.left = rect_params.left;
      obj_meta->detector_bbox_info.org_bbox_coords.top = rect_params.top;
      obj_meta->detector_bbox_info.org_bbox_coords.width = rect_params.width;
      obj_meta->detector_bbox_info.org_bbox_coords.height = rect_params.height;

      g_strlcpy (obj_meta->obj_label, text.c_str(), MAX_LABEL_SIZE);
      /* display_text requires heap allocated memory. */
      text_params.display_text = g_strdup (text.c_str());
      /* Display text above the left top corner of the object. */
      text_params.x_offset = rect_params.left;
      text_params.y_offset = rect_params.top - 10;
      /* Set black background for the text. */
      text_params.set_bg_clr = 1;
      text_params.text_bg_clr = (NvOSD_ColorParams) {
      0, 0, 0, 1};
      /* Font face, size and color. */
      text_params.font_params.font_name = "microhei";
      text_params.font_params.font_size = 11;
      text_params.font_params.font_color = (NvOSD_ColorParams) {
      1, 1, 1, 1};

      // @TODO(tylerz): currently set parent_obj_meta to be NULL
      nvds_add_obj_meta_to_frame (frame_meta, obj_meta, NULL);
      offset += 1;
    }
  }
  nvds_release_meta_lock (batch_meta);
  return TRUE;
}

#if DEBUG
static int global_id = 0;
#endif
/* Output Processing Thread */
void nvOCDRAlgorithm::OutputThread(void)
{
  GstFlowReturn flow_ret;
  GstBuffer *outBuffer = NULL;
  NvBufSurface *outSurf = NULL;
  std::unique_lock<std::mutex> lk(m_processLock);
  NvDsBatchMeta *batch_meta = NULL;

  /* Run till signalled to stop. */
  while (1) {

    /* Wait if processing queue is empty. */
    if (m_processQ.empty()) {
      if (m_stop == TRUE) {
        break;
      }
      m_processCV.wait(lk);
      continue;
    }

    PacketInfo packetInfo = m_processQ.front();
    m_processQ.pop();

    m_processCV.notify_all();
    lk.unlock();

    NvBufSurface *in_surf = getNvBufSurface (packetInfo.inbuf);
    batch_meta = gst_buffer_get_nvds_batch_meta (packetInfo.inbuf);
    // Add custom algorithm logic here
    // Once buffer processing is done, push the buffer to the downstream by using gst_pad_push function
    int num_in_meta = batch_meta->num_frames_in_batch;

    NvDsMetaList * l_frame = NULL;
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) 
    {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
      m_temp_surf.surfaceList[frame_meta->batch_id] = in_surf->surfaceList[frame_meta->batch_id];
    }
    m_temp_surf.numFilled=num_in_meta;
    m_temp_surf.memType=in_surf->memType;

    //Transform the data format from NV12 to BGR
    NvBufSurfTransform_Error err = NvBufSurfTransformError_Success;

    err = NvBufSurfTransformSetSessionParams (&m_transform_config_params);
    if (err != NvBufSurfTransformError_Success) {
        GST_ELEMENT_ERROR (m_element, STREAM, FAILED,
        ("NvBufSurfTransformSetSessionParams failed with error %d", err),
            (NULL));
        return;
    }

    err = NvBufSurfTransform (&m_temp_surf, m_process_surf, &m_transform_params);
    if (err != NvBufSurfTransformError_Success) {
        GST_ELEMENT_ERROR (m_element, STREAM, FAILED, 
          ("NvBufSurfTransform failed with error %d while converting "
          "buffer\n",err), (NULL));
        return;
    }

    void *imagedata_ptr = NULL;
    if (m_process_surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
        imagedata_ptr = (uint8_t *)m_egl_frame.frame.pPitch[0];
    } else {
        imagedata_ptr = (uint8_t *)m_process_surf->surfaceList[0].dataPtr;
    }

    // 2D copy the data to remove padding
    if (m_interbuffer != nullptr)
    {
      size_t dst_offset = 0;
      size_t dst_pitch = m_batch_width * 3;
      size_t dst_size = m_batch_height * dst_pitch;
      size_t src_offset = 0;
      size_t src_pitch = m_process_surf->surfaceList[0].pitch;
      size_t src_size = m_batch_height * src_pitch;
      for(int i = 0; i < num_in_meta; i++)
      {
        ck(cudaMemcpy2D(reinterpret_cast<uint8_t *>(m_interbuffer) + dst_offset,
                        dst_pitch, reinterpret_cast<uint8_t *>(imagedata_ptr)+src_offset,
                        src_pitch, dst_pitch, m_batch_height, cudaMemcpyDeviceToDevice));
        dst_offset += dst_size;
        src_offset += src_size;
      }
    }

    // Construct nvOCDR input
    nvOCDRInput input;
    input.device_type = GPU;
    input.shape[0] = num_in_meta;
    input.shape[1] = m_batch_height;
    input.shape[2] = m_batch_width;
    input.shape[3] = 3;
    if (m_interbuffer != nullptr)
      input.mem_ptr = m_interbuffer;
    else
      input.mem_ptr = imagedata_ptr;
    // Do nvOCDR inference
    nvOCDROutputMeta output;
    // nvOCDR_inference(input, &output, m_nvOCDRLib);
    if (m_IsHighResolution)
      nvOCDR_high_resolution_inference(input, &output, m_nvOCDRLib, m_OverlapRatio);
    else
      nvOCDR_inference(input, &output, m_nvOCDRLib);
#if DEBUG
    // dump the input image
    std::vector<uint8_t> raw_data(m_process_surf->surfaceList[0].dataSize);
    cudaMemcpy(raw_data.data(), imagedata_ptr, m_process_surf->surfaceList[0].dataSize, cudaMemcpyDeviceToHost);
    cv::Mat frame_out(m_batch_height, m_batch_width, CV_8UC3, raw_data.data(), m_process_surf->surfaceList[0].pitch);
    std::string img_path = "./debug_img/debug_" + std::to_string(global_id) + ".jpg";
    cv::imwrite(img_path, frame_out);
    global_id += 1;
    // cudaMemcpy(frame_out.data, (uchar*)imagedata_ptr + 3 * m_batch_width * m_batch_height, 3 * m_batch_width * m_batch_height, cudaMemcpyDeviceToHost);
    // cv::imwrite("./debug_1.png", frame_out);
    // Print the output text
    int offset = 0;
    printf("output batch size: %d\n", output.batch_size);
    for(int i = 0; i < output.batch_size; i++)
    {
        for(int j = 0; j < output.text_cnt[i]; j++)
        {
            printf("%d : %s, %ld\n", i, output.text_ptr[offset].ch, strlen(output.text_ptr[offset].ch));
            offset += 1;
        }
    }
#endif
    // Update meta data
    nvds_add_ocdr_meta(batch_meta, output);
    free(output.text_ptr);
    // Transform IP case
    outSurf = in_surf;
    outBuffer = packetInfo.inbuf;

    // Output buffer parameters checking
    if (outSurf->numFilled != 0)
    {
        g_assert ((guint)m_outVideoInfo.width == outSurf->surfaceList->width);
        g_assert ((guint)m_outVideoInfo.height == outSurf->surfaceList->height);
    }

    flow_ret = gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (m_element), outBuffer);

    lk.lock();
    continue;
  }
  outputthread_stopped = true;
  lk.unlock();
  return;
}

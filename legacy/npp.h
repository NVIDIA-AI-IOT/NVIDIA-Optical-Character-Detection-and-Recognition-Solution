
class NPPUtil
{
public: 
  static void Resize_8UC3(const std::string& in_name, const std::string & out_name, cudaStream_t stream) {
    BufferManager &mBufManager = BufferManager::Instance();
    auto in_dim = mBufManager.getBuf2DDim(in_name);
    auto out_dim = mBufManager.getBuf2DDim(out_name);

    LOG(INFO) << in_dim << " resize to "<< out_dim;
    // NppStreamContext ctx{stream};
    NppStatus result = nppiResize_8u_C3R_Ctx(static_cast<unsigned char*>(mBufManager.getBuffer(in_name, DEVICE)), 
                                         mBufManager.getBuf2DPitch(in_name), 
                                        {in_dim.d[3], in_dim.d[2]},
                                        {0, 0, in_dim.d[3], in_dim.d[2]}, 
                                        static_cast<unsigned char*>(mBufManager.getBuffer(out_name, DEVICE)), 
                                        mBufManager.getBuf2DPitch(out_name),
                                        {in_dim.d[3], in_dim.d[2]},
                                        {0, 0, out_dim.d[3], out_dim.d[2]}, NPPI_INTER_LINEAR, {stream});
    if (result != NPP_SUCCESS) {
      LOG(ERROR) << "Error executing Resize -- code: " << result << std::endl;
    }
  }
  // static void 

};


void nvOCDR::testfunc(cv::Mat & input_image) {
    // size_t test_size = mResizeInfo.second.width * mResizeInfo.second.height * 3;
  int h = input_image.rows;
  int w = input_image.cols;

  auto &buf_manager = BufferManager::Instance();
  buf_manager.copyHostToDevice("input_image", mStream);
  buf_manager.initBuffer2D("input_image_resized", 3, 960, 960, 1, true);
  NPPUtil::Resize_8UC3("input_image", "input_image_resized", mStream);


  // buf_manager.copyHostToDevice("input_image", mStream);
  // buf_manager.copyDeviceToHost("input_image", mStream);

  // auto o_dev = buf_manager.getBuffer("input_image", HOST);
  buf_manager.copyDeviceToHost("input_image_resized", mStream);

  cv::Mat recovery_aa(960, 960, CV_8UC3, buf_manager.getBuffer("input_image_resized", HOST), cv::Mat::AUTO_STEP);
    cudaStreamSynchronize(mStream);

  // memcpy(recovery_aa.data, o_dev, h * w * 3);
  // checkCudaErrors(cudaMemcpy2D(recovery_aa.ptr(), w * 3, o_dev, 11776, w * 3, h, cudaMemcpyDeviceToHost));
  cv::imwrite("rec2.png", recovery_aa);



  int nh = 960;
  int nw = 960;

  int image_a_pitch;
  NppiSize image_a_size = {.width = w, .height = h};
  NppiRect image_a_roi = {.x = 0, .y = 0, .width = w, .height =h};
  Npp8u* image_a = nppiMalloc_8u_C3(w,h, &image_a_pitch);
  LOG(ERROR) << "image_a_pitch : " << image_a_pitch;
  LOG(ERROR) << "input_image.step : " << input_image.step;
  checkCudaErrors(cudaMemcpy2D(image_a, image_a_pitch, input_image.data, input_image.step, w * 3, h, cudaMemcpyHostToDevice));

  // std::vector<uint8_t> recovery_buf(w * h * 3);
  cv::Mat recovery_a(h, w, CV_8UC3);
  checkCudaErrors(cudaMemcpy2D(recovery_a.ptr(), w * 3, image_a, image_a_pitch, w * 3, h, cudaMemcpyDeviceToHost));
  cv::imwrite("recoery.png", recovery_a);

  int image_b_pitch;
  NppiSize image_b_size = {.width = nw, .height = nh};
  NppiRect image_b_roi = {.x = 0, .y = 0, .width = nw, .height = nh};
  Npp8u* image_b = nppiMalloc_8u_C3(nw, nh, &image_b_pitch);
  LOG(ERROR) << "image_b_pitch : " << image_b_pitch;

  NppStatus result = nppiResize_8u_C3R(image_a, image_a_pitch, image_a_size,image_a_roi, image_b, image_b_pitch, image_b_size,image_b_roi, NPPI_INTER_LINEAR);
  if (result != NPP_SUCCESS) {
    LOG(ERROR) << "Error executing Resize -- code: " << result << std::endl;
  }
  // std::vector<uint8_t> out_buf(image_b_pitch * nh);
  cv::Mat viz(nh, nw, CV_8UC3);
  LOG(ERROR) << "viz.step : " << viz.step;
  checkCudaErrors(cudaMemcpy2D(viz.data, viz.step, image_b, image_b_pitch, nw * 3, nh, cudaMemcpyDeviceToHost));

  // for(size_t i = 0; i < 960*2; i++) {
  //   std::cout << (int)out_buf[i] << " ";
  // }
  LOG(INFO) << "";
  cv::imwrite("nppresize.png", viz);
  // cudaMemcpy2D(&out_buf[0], image_a_pitch, &buf[0], w, w, h, cudaMemcpyHostToDevice);




  // // auto a = mBufManager.getBuffer("input_image", BUFFER_TYPE::DEVICE);
  // // auto a_d = (uint8_t*)mBufManager.getBuffer("input_image", BUFFER_TYPE::DEVICE);
  // // mBufManager.initBuffer("input_image_resized", 960 * 3 * 960, true);
  // // auto a_resize = mBufManager.getBuffer("input_image_resized", BUFFER_TYPE::DEVICE);
  // // mBufManager.copyHostToDevice("input_image", mStream);
  // mBufManager.copyHostToDevice("input_image", mStream);
  // cudaStreamSynchronize(mStream);

  // // checkCudaErrors(cudaMemcpy(image_a, a, 3904 * 3904 * 3,
  // //                                   cudaMemcpyHostToDevice));

  // NppStatus result = nppiResize_8u_C3R(static_cast<unsigned char*>(mBufManager.getBuffer("input_image", BUFFER_TYPE::DEVICE)), 3904, image_a_size, image_a_roi, static_cast<unsigned char*>(image_b), image_b_pitch, image_b_size, image_b_roi, NPPI_INTER_LINEAR);
  // // cv::Mat a_img()
  

  // // LOG(INFO) << image_b_pitch / 3;
  // std::vector<uint8_t> t (960 * image_b_pitch);
  // cv::Mat output_npp(960, image_b_pitch/3, CV_8UC3, &t[0]);
  // checkCudaErrors(cudaMemcpy(&t[0],  image_b, 960 * 960 * 3, cudaMemcpyDeviceToHost));

  // for(size_t i = 0; i < 10; i ++) {
  //   LOG(ERROR) << t[i];
  //   // LOG(ERROR) << a_d[i];
  // }

  // cv::imwrite("nppreize.png", output_npp);

}
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>

/************************************************************************************
Steps for inference: 
Step 1. Load model
Step 2. Load image 
Step 3. Preprocess image
Step 4. Inference 
Step 5. Postprocess image
Step 6. Output image
  **********************************************************************************/

//Load model
string model_path = "\";
cv::dnn::Net model = cv::dnn::readNetFromONNX("Model.onnx");

//Load image 

cv::Mat image = cv::imread("path/to/your/image.jpg", cv::IMREAD_COLOR);


#include <opencv2/opencv.hpp>
#include <NvInfer.h>
//#include <cuda_runtime_api.h>
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>
int num_classes = 5; 
std::vector<cv::Rect> boxes;
std::vector<float> confidences;
std::vector<int> class_ids;
std::vector<int> all_indices;
std::vector<std::string> class_names = {
    "ControlledFire", "Entire Image Nonfire", "Fire", "Nonfire", "Smoke"
};
/************************************************************************************
Steps for inference: 
Step 1. Load model
Step 2. Load image 
Step 3. Preprocess image
Step 4. Inference 
Step 5. Postprocess image
Step 6. Output image
  **********************************************************************************/
cv::mat Preprocessing(cv::Mat frame){
	 //Preprocessing
	//bgr to rgb
	cv::Mat rgb_frame;
	cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
	//image resize
	cv::Mat correct_frame;
	int w = 640;
	int h = 640;
	cv::resize(rgb_frame, correct_frame, cv::Size(w, h));
	//normalizing frame
	cv::Mat normalized_frame;
	double alpha = 0.0; 
	double beta = 0.0;
	cv::normalize(correct_frame, normalized_frame, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::Mat blob_image; 
	blob_image = cv::dnn::blobFromImage(normalized_frame, 1.0/255.0, cv::Size(640, 640), cv::Scalar(), true, false); 
	return blob_image;
}

void Confidenceacoord(cv::Mat output){
	int num_detections = output.size[2];
	float* data = (float*)output.data;
	int num_classes = 5; 
	float x_scale = (float)frame.cols / 640;
	float y_scale = (float)frame.rows / 640;
	for (int i = 0; i < num_detections; i++) {
		float max_conf = 0;
    	int class_id = 0;
    	for (int c = 0; c < num_classes; c++) {
    		float score = data[(4 + c) * num_detections + i];
    		if (score > max_conf) {
    		    max_conf = score;
    		    class_id = c;
    		}
    	}
    	if (max_conf < 0.5) continue;
    
    	float cx = data[0 * num_detections + i];
    	float cy = data[1 * num_detections + i];
    	float w  = data[2 * num_detections + i];
    	float h  = data[3 * num_detections + i];
    	int x = (int)((cx - w / 2) * x_scale);
    	int y = (int)((cy - h / 2) * y_scale);
    	int width = (int)(w * x_scale);
    	int height = (int)(h * y_scale);
    	boxes.push_back(cv::Rect(x, y, width, height));
    	confidences.push_back(max_conf);
    	class_ids.push_back(class_id);
	}
}

void NMS(int num_classes){
	for (int c = 0; c < num_classes; c++) {
    	std::vector<cv::Rect> class_boxes;
    	std::vector<float> class_confs;
    	std::vector<int> original_idx;
    	for (int i = 0; i < boxes.size(); i++) {
    	    if (class_ids[i] == c) {
    	        class_boxes.push_back(boxes[i]);
    	        class_confs.push_back(confidences[i]);
    	        original_idx.push_back(i);
    	    }
    	}
    	std::vector<int> indices;
    	cv::dnn::NMSBoxes(class_boxes, class_confs, 0.5, 0.4, indices);
    	for (int idx : indices) {
    	    all_indices.push_back(original_idx[idx]);
    	}
	}
}
int main(){
	//Load model
	//std::string model_path = "/media/nvidia/0051-D5A7/yolo11n.onnx";
	//cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
	//net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	//net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	//Load image
	cv::VideoCapture cap(10);
	// for static images : cv::Mat image = cv::imread("path/to/your/image.jpg", cv::IMREAD_COLOR);
	if (!cap.isOpened()) {
		std::cerr << "Error: Could not open camera." << std::endl;
		return -1;
	}
	auto prev_time = std::chrono::high_resolution_clock::now();
	while(true){
		cv::Mat frame;
		bool success = cap.read(frame);
		if (!success) {
            std::cerr << "Error: Could not read frame from camera." << std::endl;
            break;
        }
		//Preprocessing
		cv::Mat pre_frame = Preprocessing(frame);
		//inference
		//net.setInput(blob_image);
		//cv::Mat output = net.forward();

		//Postprocess
		//make the cx, cy, w, h, into x1, x2, y1, y2 & confidence checking
		Confidenceacoord(output);
		//NMS 
		NMS(num_classes); 
		// Draw Bounding box

		for (int idx : all_indices) {
		    cv::Rect box = boxes[idx];
		    float conf = confidences[idx];
		    int cls = class_ids[idx];
    
		    cv::rectangle(pre_frame, box, cv::Scalar(0, 0, 255), 2);
		    cv::putText(frame, class_names[cls] + " " + std::to_string(conf),
		                box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
		}
		for (int idx : all_indices) {
        	cv::rectangle(pre_frame, boxes[idx], cv::Scalar(0, 0, 255), 2);
                cv::putText(pre_frame, 
        	            class_names[class_ids[idx]],
        	            boxes[idx].tl(), 
        	            cv::FONT_HERSHEY_SIMPLEX, 
        	            0.5, 
        	            cv::Scalar(0, 0, 255), 
        	            2);
	    }
	    auto curr_time = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(curr_time - prev_time).count();
        double fps = 1000.0 / ms;
        prev_time = curr_time;
	    cv::putText(pre_frame, "FPS: " + std::to_string((int)fps), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
	    cv::imshow("Fire Detection", pre_frame);
	    if (cv::waitKey(1) == 'q') break;
	}
	cap.release();
	cv::destroyAllWindows();
	return 0;
}

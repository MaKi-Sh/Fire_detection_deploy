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
string model_path = "\media\nvidia\0051-DA57\yolo11n.onnx";
string model_name = "yolo11n.onnx";
cv::dnn::Net model = cv::dnn::readNetFromONNX(model_name);

//Load image 
cv::VideoCapture cap(10);
// for static images : cv::Mat image = cv::imread("path/to/your/image.jpg", cv::IMREAD_COLOR);
if (!cap.isOpened()) {
    std::cerr << "Error: Could not open camera." << std::endl;
    return -1;
}

int main(){
	while(true){
		cv::Mat frame;
		bool success = cap.read(frame);
		if (!success) {
            std::cerr << "Error: Could not read frame from camera." << std::endl;
            break;
        }
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
		cv::dnn:blobFromImage(normalized_frame, blob_image); 

		//inference 
		net.setInput(blob_image);
		cv::Mat final = net.forward();

		//Postprocess
		//make the cx, cy, w, h, into x1, x2, y1, y2 & confidence checking
		std::vector<cv::Rect> boxes;
		std::vector<float> confidences;
		std::vector<int> all_indicies;
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
		//NMS 
		std::vector<int> all_indices;
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
		// Draw Bounding boxes
		std::vector<std::string> class_names = {
		    "ControlledFire", "Entire Image Nonfire", "Fire", "Nonfire", "Smoke"
		};

		for (int idx : all_indices) {
		    cv::Rect box = boxes[idx];
		    float conf = confidences[idx];
		    int cls = class_ids[idx];
    
		    cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2);
		    cv::putText(frame, class_names[cls] + " " + std::to_string(conf),
		                box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
		}
		for (int idx : all_indices) {
        	cv::rectangle(frame, boxes[idx], cv::Scalar(0, 0, 255), 2);
        	cv::putText(frame, 
        	            class_names[class_ids[idx]],
        	            boxes[idx].tl(), 
        	            cv::FONT_HERSHEY_SIMPLEX, 
        	            0.5, 
        	            cv::Scalar(0, 0, 255), 
        	            2);
	    }
	    cv::imshow("Fire Detection", frame);
	    if (cv::waitKey(1) == 'q') break;
	}
	cap.release(); 
	cv::destroyAllWindows();
}

//Preprocessing 

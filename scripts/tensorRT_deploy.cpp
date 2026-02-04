#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
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

// TensorRT Logger
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;

/************************************************************************************
Steps for inference:
Step 1. Load model
Step 2. Load image
Step 3. Preprocess image
Step 4. Inference
Step 5. Postprocess image
Step 6. Output image
  **********************************************************************************/
cv::Mat Preprocessing(cv::Mat frame, float* gpu_input, int input_size){
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
	cv::Mat float_frame;
	correct_frame.convertTo(float_frame, CV_32FC3, 1.0 / 255.0);

	// Convert HWC to CHW format for TensorRT
	std::vector<cv::Mat> channels(3);
	cv::split(float_frame, channels);
	std::vector<float> input_data(input_size);
	int channel_size = 640 * 640;
	memcpy(input_data.data(), channels[0].data, channel_size * sizeof(float));
	memcpy(input_data.data() + channel_size, channels[1].data, channel_size * sizeof(float));
	memcpy(input_data.data() + 2 * channel_size, channels[2].data, channel_size * sizeof(float));

	// Copy to GPU
	cudaMemcpy(gpu_input, input_data.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);

	return correct_frame;
}

void Confidenceacoord(float* output_data, int num_detections, cv::Mat& frame){
	float x_scale = (float)frame.cols / 640;
	float y_scale = (float)frame.rows / 640;
	for (int i = 0; i < num_detections; i++) {
		float max_conf = 0;
    	int class_id = 0;
    	for (int c = 0; c < num_classes; c++) {
    		float score = output_data[(4 + c) * num_detections + i];
    		if (score > max_conf) {
    		    max_conf = score;
    		    class_id = c;
    		}
    	}
    	if (max_conf < 0.5) continue;

    	float cx = output_data[0 * num_detections + i];
    	float cy = output_data[1 * num_detections + i];
    	float w  = output_data[2 * num_detections + i];
    	float h  = output_data[3 * num_detections + i];
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
    	for (size_t i = 0; i < boxes.size(); i++) {
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
	// Load TensorRT engine
	std::string engine_path = "/media/nvidia/0051-D5A7/yolo11n.engine";
	std::ifstream engine_file(engine_path, std::ios::binary);
	if (!engine_file) {
		std::cerr << "Error: Could not open engine file: " << engine_path << std::endl;
		return -1;
	}
	engine_file.seekg(0, std::ios::end);
	size_t engine_size = engine_file.tellg();
	engine_file.seekg(0, std::ios::beg);
	std::vector<char> engine_data(engine_size);
	engine_file.read(engine_data.data(), engine_size);
	engine_file.close();

	// Create runtime and deserialize engine
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
	nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_size);
	if (!engine) {
		std::cerr << "Error: Failed to deserialize engine" << std::endl;
		return -1;
	}
	nvinfer1::IExecutionContext* context = engine->createExecutionContext();
	if (!context) {
		std::cerr << "Error: Failed to create execution context" << std::endl;
		return -1;
	}

	// Allocate GPU memory for input and output
	int input_size = 1 * 3 * 640 * 640;
	int output_size = 1 * (4 + num_classes) * 8400; // YOLOv11 output: [1, 9, 8400]

	float* gpu_input;
	float* gpu_output;
	cudaMalloc((void**)&gpu_input, input_size * sizeof(float));
	cudaMalloc((void**)&gpu_output, output_size * sizeof(float));

	void* bindings[] = {gpu_input, gpu_output};

	// Host output buffer
	std::vector<float> output_data(output_size);

	//Load image
	cv::VideoCapture cap(10);
	// for static images : cv::Mat image = cv::imread("path/to/your/image.jpg", cv::IMREAD_COLOR);
	if (!cap.isOpened()) {
		std::cerr << "Error: Could not open camera." << std::endl;
		cudaFree(gpu_input);
		cudaFree(gpu_output);
		delete context;
		delete engine;
		delete runtime;
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

		// Clear vectors for each frame
		boxes.clear();
		confidences.clear();
		class_ids.clear();
		all_indices.clear();

		//Preprocessing
		cv::Mat display_frame = frame.clone();
		Preprocessing(frame, gpu_input, input_size);

		//TensorRT inference
		context->executeV2(bindings);

		// Copy output from GPU to CPU
		cudaMemcpy(output_data.data(), gpu_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

		//Postprocess
		//make the cx, cy, w, h, into x1, x2, y1, y2 & confidence checking
		int num_detections = 8400;
		Confidenceacoord(output_data.data(), num_detections, frame);
		//NMS
		NMS(num_classes);
		// Draw Bounding box

		for (int idx : all_indices) {
		    cv::Rect box = boxes[idx];
		    float conf = confidences[idx];
		    int cls = class_ids[idx];

		    cv::rectangle(display_frame, box, cv::Scalar(0, 0, 255), 2);
		    cv::putText(display_frame, class_names[cls] + " " + std::to_string(conf),
		                box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
		}
	    auto curr_time = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(curr_time - prev_time).count();
        double fps = 1000.0 / ms;
        prev_time = curr_time;
	    cv::putText(display_frame, "FPS: " + std::to_string((int)fps), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
	    cv::imshow("Fire Detection", display_frame);
	    if (cv::waitKey(1) == 'q') break;
	}

	// Cleanup
	cap.release();
	cv::destroyAllWindows();
	cudaFree(gpu_input);
	cudaFree(gpu_output);
	delete context;
	delete engine;
	delete runtime;
	return 0;
}

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>

// TensorRT Logger
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

// Structure to hold TensorRT inference context
struct TensorRTContext {
    Logger logger;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* exec_context = nullptr;
    float* gpu_input = nullptr;
    float* gpu_output = nullptr;
    float* pinned_input = nullptr;   // Pinned host memory for faster transfers
    float* pinned_output = nullptr;  // Pinned host memory for faster transfers
    cudaStream_t stream = nullptr;   // CUDA stream for async operations
    void* bindings[2] = {nullptr, nullptr};
    std::vector<float> output_data;
    int input_size = 1 * 3 * 640 * 640;
    int output_size = 0;
    int num_classes = 5;
    std::vector<std::string> class_names = {
        "ControlledFire", "Entire Image Nonfire", "Fire", "Nonfire", "Smoke"
    };
};

// Structure to hold detection results
struct DetectionResult {
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<int> indices;
};

/************************************************************************************
Steps for inference:
Step 1. Load models         -> initTensorRT()
Step 2. Load image          -> (user handles this)
Step 3. Preprocess image    -> preprocessFrame()
Step 4. Inference           -> runInference()
Step 5. Postprocess image   -> postprocessDetections()
Step 6. Output image        -> drawDetections()
Step 7. Cleanup             -> cleanupTensorRT()
************************************************************************************/

// Initialize TensorRT engine from file
bool initTensorRT(TensorRTContext& ctx, const std::string& engine_path) {
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file) {
        std::cerr << "Error: Could not open engine file: " << engine_path << std::endl;
        return false;
    }
    engine_file.seekg(0, std::ios::end);
    size_t engine_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(engine_size);
    engine_file.read(engine_data.data(), engine_size);
    engine_file.close();

    ctx.runtime = nvinfer1::createInferRuntime(ctx.logger);
    ctx.engine = ctx.runtime->deserializeCudaEngine(engine_data.data(), engine_size);
    if (!ctx.engine) {
        std::cerr << "Error: Failed to deserialize engine" << std::endl;
        return false;
    }
    ctx.exec_context = ctx.engine->createExecutionContext();
    if (!ctx.exec_context) {
        std::cerr << "Error: Failed to create execution context" << std::endl;
        return false;
    }

    // Allocate GPU memory
    ctx.output_size = 1 * (4 + ctx.num_classes) * 8400;
    cudaMalloc(&ctx.gpu_input, ctx.input_size * sizeof(float));
    cudaMalloc(&ctx.gpu_output, ctx.output_size * sizeof(float));
    ctx.bindings[0] = ctx.gpu_input;
    ctx.bindings[1] = ctx.gpu_output;
    ctx.output_data.resize(ctx.output_size);

    // Allocate pinned memory for faster CPU-GPU transfers
    cudaMallocHost(&ctx.pinned_input, ctx.input_size * sizeof(float));
    cudaMallocHost(&ctx.pinned_output, ctx.output_size * sizeof(float));

    // Create CUDA stream for async operations
    cudaStreamCreate(&ctx.stream);

    return true;
}

// Cleanup TensorRT resources
void cleanupTensorRT(TensorRTContext& ctx) {
    // Synchronize stream before cleanup
    if (ctx.stream) {
        cudaStreamSynchronize(ctx.stream);
        cudaStreamDestroy(ctx.stream);
        ctx.stream = nullptr;
    }

    // Free pinned memory
    if (ctx.pinned_input) cudaFreeHost(ctx.pinned_input);
    if (ctx.pinned_output) cudaFreeHost(ctx.pinned_output);
    ctx.pinned_input = nullptr;
    ctx.pinned_output = nullptr;

    // Free GPU memory
    if (ctx.gpu_input) cudaFree(ctx.gpu_input);
    if (ctx.gpu_output) cudaFree(ctx.gpu_output);
    ctx.gpu_input = nullptr;
    ctx.gpu_output = nullptr;

    // Destroy TensorRT objects
    if (ctx.exec_context) delete ctx.exec_context;
    if (ctx.engine) delete ctx.engine;
    if (ctx.runtime) delete ctx.runtime;
    ctx.exec_context = nullptr;
    ctx.engine = nullptr;
    ctx.runtime = nullptr;
}

// Preprocess frame for inference (optimized with pinned memory)
cv::Mat preprocessFrame(const cv::Mat& frame, TensorRTContext& ctx) {
	// BGR to RGB
	cv::Mat rgb_frame;
	cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);

	// Resize to 640x640
	cv::Mat correct_frame;
	cv::resize(rgb_frame, correct_frame, cv::Size(640, 640));

	// Normalize to [0, 1]
	cv::Mat float_frame;
	correct_frame.convertTo(float_frame, CV_32FC3, 1.0 / 255.0);

	// Convert HWC to CHW format directly into pinned memory
	std::vector<cv::Mat> channels(3);
	cv::split(float_frame, channels);
	constexpr int channel_size = 640 * 640;
	memcpy(ctx.pinned_input, channels[0].data, channel_size * sizeof(float));
	memcpy(ctx.pinned_input + channel_size, channels[1].data, channel_size * sizeof(float));
	memcpy(ctx.pinned_input + 2 * channel_size, channels[2].data, channel_size * sizeof(float));

	// Async copy to GPU using CUDA stream
	cudaMemcpyAsync(ctx.gpu_input, ctx.pinned_input, ctx.input_size * sizeof(float),
	                cudaMemcpyHostToDevice, ctx.stream);

	return correct_frame;
}

// Run TensorRT inference (optimized with CUDA stream)
void runInference(TensorRTContext& ctx) {
	// Execute inference on stream
	ctx.exec_context->enqueueV2(ctx.bindings, ctx.stream, nullptr);

	// Async copy output to pinned memory
	cudaMemcpyAsync(ctx.pinned_output, ctx.gpu_output, ctx.output_size * sizeof(float),
	                cudaMemcpyDeviceToHost, ctx.stream);

	// Synchronize stream to ensure results are ready
	cudaStreamSynchronize(ctx.stream);

	// Copy from pinned memory to output_data vector
	memcpy(ctx.output_data.data(), ctx.pinned_output, ctx.output_size * sizeof(float));
}

// Parse detections from output (confidence thresholding + coordinate conversion)
void parseDetections(float* output_data, int num_detections, const cv::Mat& frame,
                     int num_classes, DetectionResult& result, float conf_threshold = 0.5f) {
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
		if (max_conf < conf_threshold) continue;

		float cx = output_data[0 * num_detections + i];
		float cy = output_data[1 * num_detections + i];
		float w  = output_data[2 * num_detections + i];
		float h  = output_data[3 * num_detections + i];
		int x = (int)((cx - w / 2) * x_scale);
		int y = (int)((cy - h / 2) * y_scale);
		int width = (int)(w * x_scale);
		int height = (int)(h * y_scale);
		result.boxes.push_back(cv::Rect(x, y, width, height));
		result.confidences.push_back(max_conf);
		result.class_ids.push_back(class_id);
	}
}

// Apply Non-Maximum Suppression
void applyNMS(DetectionResult& result, int num_classes, float nms_threshold = 0.4f) {
	for (int c = 0; c < num_classes; c++) {
		std::vector<cv::Rect> class_boxes;
		std::vector<float> class_confs;
		std::vector<int> original_idx;
		for (size_t i = 0; i < result.boxes.size(); i++) {
			if (result.class_ids[i] == c) {
				class_boxes.push_back(result.boxes[i]);
				class_confs.push_back(result.confidences[i]);
				original_idx.push_back(i);
			}
		}
		std::vector<int> indices;
		cv::dnn::NMSBoxes(class_boxes, class_confs, 0.5f, nms_threshold, indices);
		for (int idx : indices) {
			result.indices.push_back(original_idx[idx]);
		}
	}
}

// Full postprocessing: parse detections + NMS
DetectionResult postprocessDetections(TensorRTContext& ctx, const cv::Mat& frame,
                                       float conf_threshold = 0.5f, float nms_threshold = 0.4f) {
	DetectionResult result;
	int num_detections = 8400;
	parseDetections(ctx.output_data.data(), num_detections, frame, ctx.num_classes, result, conf_threshold);
	applyNMS(result, ctx.num_classes, nms_threshold);
	return result;
}

// Draw detection boxes on frame
void drawDetections(cv::Mat& frame, const DetectionResult& result,
                    const std::vector<std::string>& class_names) {
	for (int idx : result.indices) {
		cv::Rect box = result.boxes[idx];
		float conf = result.confidences[idx];
		int cls = result.class_ids[idx];

		cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2);
		std::string label = class_names[cls] + " " + std::to_string(conf).substr(0, 4);
		cv::putText(frame, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
	}
}

// Draw FPS on frame
void drawFPS(cv::Mat& frame, double fps) {
	cv::putText(frame, "FPS: " + std::to_string((int)fps), cv::Point(10, 30),
	            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
}

// Convenience function: process a single frame end-to-end
DetectionResult processFrame(const cv::Mat& frame, TensorRTContext& ctx,
                              float conf_threshold, float nms_threshold) {
	preprocessFrame(frame, ctx);
	runInference(ctx);
	return postprocessDetections(ctx, frame, conf_threshold, nms_threshold);
}

/* Example usage in main.cpp:

#include "tensorRT_deploy.cpp"

int main() {
    TensorRTContext ctx;

    // Initialize TensorRT
    if (!initTensorRT(ctx, "/media/nvidia/0051-D5A7/yolo11n.engine")) {
        return -1;
    }

    // Open camera
    cv::VideoCapture cap(10);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        cleanupTensorRT(ctx);
        return -1;
    }

    auto prev_time = std::chrono::high_resolution_clock::now();
    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cerr << "Error: Could not read frame." << std::endl;
            break;
        }

        // Process frame and get detections
        DetectionResult result = processFrame(frame, ctx);

        // Draw results
        cv::Mat display_frame = frame.clone();
        drawDetections(display_frame, result, ctx.class_names);

        // Calculate and draw FPS
        auto curr_time = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(curr_time - prev_time).count();
        prev_time = curr_time;
        drawFPS(display_frame, 1000.0 / ms);

        cv::imshow("Fire Detection", display_frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    cleanupTensorRT(ctx);
    return 0;
}
*/

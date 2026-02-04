/*********************************************
  Engine Building phase 
  1. Create a network definition
  2. Specify a configuration for the builder 
  3. Call the builder to create the engine 
 ********************************************/ 
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <iostream>
#include <fstream>
using namespace std;
using namespace nvonnxparser;
using namespace nvinfer1;
class Logger : public ILogger
{
	void log(Severity severity, const char* msg) noexcept override
	{
		// suppress info-level messages
		if (severity <= Severity::kWARNING)
			std::cout << msg << std::endl;
	}
} logger;

int main(){
	//Model path
	string model_path = "/media/nvidia/0051-D5A7/yolo11n.onnx";
	//Network definition
	//Creating builder
	IBuilder* builder = createInferBuilder(logger);
	//Create network  
	uint32_t flag = 0;  // 0 means explicit batch mode in newer TensorRT
	INetworkDefinition* network = builder->createNetworkV2(flag); 
	//Load ONNX model
	IParser* parser = createParser(*network, logger);
	parser->parseFromFile(model_path.c_str(),
			static_cast<int32_t>(ILogger::Severity::kWARNING));
	for (int32_t i = 0; i < parser->getNbErrors(); ++i){
		std::cout << parser->getError(i)->desc() << std::endl;
	}

	//Configure and build
	IBuilderConfig* config = builder->createBuilderConfig();
	// Set workspace memory to 256 MB (increase if needed for larger models)
	config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 256U << 20);

	// Enable FP16 mode for Jetson devices (reduces memory and improves performance)
	if (builder->platformHasFastFp16()) {
		std::cout << "FP16 supported, enabling FP16 mode" << std::endl;
		config->setFlag(BuilderFlag::kFP16);
	}

	std::cout << "Building TensorRT engine... This may take a few minutes." << std::endl;
	IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);

	// Check if build succeeded
	if (!serializedModel) {
		std::cerr << "ERROR: Failed to build TensorRT engine!" << std::endl;
		delete parser;
		delete network;
		delete config;
		delete builder;
		return 1;
	}

	std::cout << "Engine built successfully! Size: " << serializedModel->size() << " bytes" << std::endl;

	//Save model to USB drive
	string engine_path = "/media/nvidia/0051-D5A7/yolo11n.engine";
	ofstream engineFile(engine_path, std::ios::binary);
	engineFile.write(reinterpret_cast<const char*>(serializedModel->data()),
                 serializedModel->size());
	engineFile.close();

	std::cout << "Engine saved to " << engine_path << std::endl;

	//Clean up
	delete parser;
	delete network;
	delete config;
	delete builder;
	delete serializedModel;

	return 0;
}

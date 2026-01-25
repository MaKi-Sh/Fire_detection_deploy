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
	config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20);
	config->setMemoryPoolLimit(MemoryPoolType::kTACTIC_SHARED_MEMORY, 48 << 10);
	IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
	
	//Save model
	ofstream engineFile("yolo11n.engine", std::ios::binary);
	engineFile.write(reinterpret_cast<const char*>(serializedModel->data()), 
                 serializedModel->size());
	engineFile.close();

	//Clean up 
	delete parser;
	delete network;
	delete config;
	delete builder;
	delete serializedModel; 
}

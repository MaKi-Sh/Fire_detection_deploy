#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <NvInfer.h>
#include <NvOnnxParser.h>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

bool buildEngine(const std::string& onnxPath, const std::string& enginePath, Logger& logger) {
    std::cout << "\nProcessing: " << onnxPath << std::endl;

    // Create builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    if (!builder) {
        std::cerr << "ERROR: Failed to create InferBuilder!" << std::endl;
        return false;
    }

    // Create network (explicit batch)
    const auto explicitBatch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    if (!network) {
        std::cerr << "ERROR: Failed to create network!" << std::endl;
        delete builder;
        return false;
    }

    // Create ONNX parser
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    if (!parser) {
        std::cerr << "ERROR: Failed to create ONNX parser!" << std::endl;
        delete network;
        delete builder;
        return false;
    }

    // Parse ONNX model
    if (!parser->parseFromFile(onnxPath.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "ERROR: Failed to parse ONNX model from file: " << onnxPath << std::endl;
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // Create builder config
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    if (!config) {
        std::cerr << "ERROR: Failed to create builder config!" << std::endl;
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // Set memory pool limit (1 GiB)
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    // Enable FP16 if supported
    if (builder->platformHasFastFp16()) {
        std::cout << "FP16 supported, enabling FP16 mode" << std::endl;
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // Build serialized engine
    std::cout << "Building TensorRT engine... This may take a few minutes." << std::endl;
    nvinfer1::IHostMemory* serializedEngine = builder->buildSerializedNetwork(*network, *config);
    if (!serializedEngine) {
        std::cerr << "ERROR: Failed to build TensorRT engine!" << std::endl;
        delete config;
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    std::cout << "Engine built successfully! Size: " << serializedEngine->size() << " bytes" << std::endl;

    // Save engine to file
    std::ofstream engineFile(enginePath, std::ios::binary);
    if (!engineFile) {
        std::cerr << "ERROR: Could not open file for writing: " << enginePath << std::endl;
        delete serializedEngine;
        delete config;
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    engineFile.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    engineFile.close();
    std::cout << "Engine saved to " << enginePath << std::endl;

    // Cleanup
    delete serializedEngine;
    delete config;
    delete parser;
    delete network;
    delete builder;

    return true;
}

int main() {
    Logger logger;

    const std::string modelDir = "/media/nvidia/0051-D5A7";

    std::vector<std::string> modelNames = {
        "Fire-detect_yolo11m_baseline_best",
        "Fire-detect_yolo11m_fire-recall_opt_best",
        "Fire-detect_yolo11m_recall_opt_best",
        "Fire-detect_yolo11n_baseline_best",
        "Fire-detect_yolo11n_fire-recall_opt_best",
        "Fire-detect_yolo11n_recall_opt_best",
        "Fire-detect_yolo11s_baseline_best",
        "Fire-detect_yolo11s_fire-recall_opt_best",
        "Fire-detect_yolo11s_recall_opt_best",
    };

    int success = 0;
    int failed = 0;

    for (const auto& name : modelNames) {
        std::string onnxPath = modelDir + "/" + name + ".onnx";
        std::string enginePath = modelDir + "/" + name + ".engine";

        if (buildEngine(onnxPath, enginePath, logger)) {
            success++;
        } else {
            std::cerr << "FAILED: " << name << std::endl;
            failed++;
        }
    }

    std::cout << "\n============================================" << std::endl;
    std::cout << "Results: " << success << " succeeded, " << failed << " failed" << std::endl;
    std::cout << "============================================" << std::endl;

    return (failed > 0) ? 1 : 0;
}

#include "tensorRT_deploy.cpp"
#include "IR_camera_detection.cpp"

// Fire class index in YOLO model (based on class_names in TensorRTContext)
constexpr int FIRE_CLASS_ID = 2;        // "Fire"
constexpr int CONTROLLED_FIRE_ID = 0;   // "ControlledFire"
constexpr int SMOKE_CLASS_ID = 4;       // "Smoke"

// Check if a detection is a fire-related class
bool isFireClass(int classId) {
    return classId == FIRE_CLASS_ID || classId == CONTROLLED_FIRE_ID;
}

// Verify fire detection by checking if hot pixels from IR camera
// fall within YOLO fire bounding boxes
bool verifyFireWithIR(const DetectionResult& yoloResult,
                      const IRDetection& irDetection,
                      const cv::Size& irFrameSize,
                      const cv::Size& rgbFrameSize,
                      float tempThreshold) {

    // Get only fire-class bounding boxes (using NMS indices)
    std::vector<cv::Rect> fireBoxes;
    for (int idx : yoloResult.indices) {
        if (isFireClass(yoloResult.class_ids[idx])) {
            fireBoxes.push_back(yoloResult.boxes[idx]);
        }
    }

    if (fireBoxes.empty()) {
        return false;  // No fire detected by YOLO
    }

    if (!irDetection.info.detected) {
        return false;  // No hot clusters detected by IR
    }

    // Check if any hot pixel above threshold falls within a fire bounding box
    for (const auto& cluster : irDetection.clusters) {
        for (const Pixel& pix : cluster) {
            if (pix.temp < tempThreshold) continue;

            // Scale IR coordinates to RGB frame coordinates
            cv::Point scaledPoint = scalePixelCoords(pix, irFrameSize, rgbFrameSize);

            // Check against all fire bounding boxes
            for (const cv::Rect& box : fireBoxes) {
                if (isPointInBox(scaledPoint.x, scaledPoint.y, box)) {
                    return true;  // Confirmed fire!
                }
            }
        }
    }

    return false;
}

// Alternative main loop: IR-first approach
// Checks IR camera first, only runs YOLO if hot spots detected
// This can save computation when no heat sources are present
/*
void mainLoopIRFirst(TensorRTContext& ctx,
                     cv::VideoCapture& rgbCap,
                     cv::VideoCapture& irCap,
                     bool useIR,
                     float confidenceThreshold,
                     float nmsThreshold,
                     float irTempThreshold,
                     int minClusterSize) {

    auto prevTime = std::chrono::high_resolution_clock::now();

    while (true) {
        cv::Mat rgbFrame;
        if (!rgbCap.read(rgbFrame)) {
            std::cerr << "Error: Could not read RGB frame" << std::endl;
            break;
        }

        IRDetection irDetection;
        cv::Size irFrameSize(0, 0);
        bool hotSpotsDetected = false;

        // Step 1: Process IR frame FIRST (if available)
        if (useIR) {
            cv::Mat irRawFrame;
            if (irCap.read(irRawFrame)) {
                irFrameSize = irRawFrame.size();
                cv::Mat irTempFrame = convertToTemperature(irRawFrame);
                irDetection = fireDetect(irTempFrame, irTempThreshold, minClusterSize);
                hotSpotsDetected = irDetection.info.detected;
            }
        }

        // Step 2: Only run YOLO if IR detected hot spots (or if IR not available)
        DetectionResult yoloResult;
        bool fireConfirmed = false;

        if (!useIR || hotSpotsDetected) {
            // Run YOLO inference on RGB frame
            yoloResult = processFrame(rgbFrame, ctx, confidenceThreshold, nmsThreshold);

            // Step 3: Verify fire detection
            if (useIR && irFrameSize.width > 0) {
                // IR + YOLO verification
                fireConfirmed = verifyFireWithIR(yoloResult, irDetection,
                                                  irFrameSize, rgbFrame.size(),
                                                  irTempThreshold);
            } else {
                // RGB-only mode: check if YOLO detected fire class
                for (int idx : yoloResult.indices) {
                    if (isFireClass(yoloResult.class_ids[idx])) {
                        fireConfirmed = true;
                        break;
                    }
                }
            }
        }

        // Draw results
        cv::Mat displayFrame = rgbFrame.clone();
        if (!useIR || hotSpotsDetected) {
            drawDetections(displayFrame, yoloResult, ctx.class_names);
        }

        // Draw fire alert if confirmed
        if (fireConfirmed) {
            cv::putText(displayFrame, "FIRE DETECTED!", cv::Point(10, 70),
                        cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
            std::cout << "FIRE DETECTED!" << std::endl;
        }

        // Calculate and draw FPS
        auto currTime = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(currTime - prevTime).count();
        prevTime = currTime;
        drawFPS(displayFrame, 1000.0 / ms);

        // Show IR info if available
        if (useIR && irDetection.info.maxTemp > 0) {
            std::string irInfo = "IR Max: " + std::to_string(static_cast<int>(irDetection.info.maxTemp)) + "C";
            cv::putText(displayFrame, irInfo, cv::Point(10, 110),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 128, 0), 2);
        }

        cv::imshow("Fire Detection", displayFrame);
        if (cv::waitKey(1) == 'q') break;
    }
}
*/

int main() {
    // Configuration
    const std::string enginePath = "/media/nvidia/0051-D5A7/yolo11n.engine";
    const int rgbCameraId = 10;
    const int irCameraId = 11;  // Adjust as needed for your IR camera

    const float confidenceThreshold = 0.5f;
    const float nmsThreshold = 0.4f;
    const float irTempThreshold = 100.0f;  // Celsius
    const int minClusterSize = 20;

    // Initialize TensorRT
    TensorRTContext ctx;
    if (!initTensorRT(ctx, enginePath)) {
        std::cerr << "Failed to initialize TensorRT engine" << std::endl;
        return -1;
    }

    // Open RGB camera
    cv::VideoCapture rgbCap(rgbCameraId);
    if (!rgbCap.isOpened()) {
        std::cerr << "Error: Could not open RGB camera " << rgbCameraId << std::endl;
        cleanupTensorRT(ctx);
        return -1;
    }

    // Open IR camera
    cv::VideoCapture irCap(irCameraId);
    if (!irCap.isOpened()) {
        std::cerr << "Warning: Could not open IR camera " << irCameraId << std::endl;
        std::cerr << "Running in RGB-only mode (no IR verification)" << std::endl;
    }

    bool useIR = irCap.isOpened();
    auto prevTime = std::chrono::high_resolution_clock::now();

    std::cout << "Fire Detection System Started" << std::endl;
    std::cout << "Press 'q' to quit" << std::endl;

    while (true) {
        // Read RGB frame
        cv::Mat rgbFrame;
        if (!rgbCap.read(rgbFrame)) {
            std::cerr << "Error: Could not read RGB frame" << std::endl;
            break;
        }

        // Run YOLO inference
        DetectionResult yoloResult = processFrame(rgbFrame, ctx, confidenceThreshold, nmsThreshold);

        // Process IR frame if available
        IRDetection irDetection;
        cv::Size irFrameSize(0, 0);

        cv::Mat irHeatmap;  // For thermal display
        if (useIR) {
            cv::Mat irRawFrame;
            if (irCap.read(irRawFrame)) {
                irFrameSize = irRawFrame.size();
                cv::Mat irTempFrame = convertToTemperature(irRawFrame);
                irDetection = fireDetect(irTempFrame, irTempThreshold, minClusterSize);

                // Create thermal heatmap visualization
                cv::Mat display;
                cv::normalize(irTempFrame, display, 0, 255, cv::NORM_MINMAX);
                display.convertTo(display, CV_8U);
                cv::applyColorMap(display, irHeatmap, cv::COLORMAP_INFERNO);
                // Scale up from 160x120 to match RGB better
                cv::resize(irHeatmap, irHeatmap, cv::Size(640, 480));
            }
        }

        // Check for confirmed fire
        bool fireConfirmed = false;
        if (useIR && irFrameSize.width > 0) {
            fireConfirmed = verifyFireWithIR(yoloResult, irDetection,
                                              irFrameSize, rgbFrame.size(),
                                              irTempThreshold);
        } else {
            // RGB-only mode: check if YOLO detected fire class
            for (int idx : yoloResult.indices) {
                if (isFireClass(yoloResult.class_ids[idx])) {
                    fireConfirmed = true;
                    break;
                }
            }
        }

        // Draw results
        cv::Mat displayFrame = rgbFrame.clone();
        drawDetections(displayFrame, yoloResult, ctx.class_names);

        // Draw fire alert if confirmed
        if (fireConfirmed) {
            cv::putText(displayFrame, "FIRE DETECTED!", cv::Point(10, 70),
                        cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
            std::cout << "FIRE DETECTED!" << std::endl;
        }

        // Calculate and draw FPS
        auto currTime = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(currTime - prevTime).count();
        prevTime = currTime;
        drawFPS(displayFrame, 1000.0 / ms);

        // Show IR info if available
        if (useIR && irDetection.info.maxTemp > 0) {
            std::string irInfo = "IR Max: " + std::to_string(static_cast<int>(irDetection.info.maxTemp)) + "C";
            cv::putText(displayFrame, irInfo, cv::Point(10, 110),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 128, 0), 2);
        }

        cv::imshow("Fire Detection", displayFrame);

        // Display thermal heatmap in separate window if available
        if (useIR && !irHeatmap.empty()) {
            cv::imshow("Thermal", irHeatmap);
        }

        if (cv::waitKey(1) == 'q') break;
    }

    // Cleanup
    rgbCap.release();
    if (useIR) irCap.release();
    cv::destroyAllWindows();
    cleanupTensorRT(ctx);

    return 0;
}

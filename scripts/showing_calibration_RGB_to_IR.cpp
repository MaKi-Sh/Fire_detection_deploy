#include "tensorRT_deploy.cpp"
#include "IR_camera_detection.cpp"

// Detect ControlledFire (class 0) since we demo with a candle
constexpr int CONTROLLED_FIRE_ID = 0;

int main() {
    // Configuration
    const std::string enginePath = "/media/nvidia/0051-D5A7/yolo11n.engine";
    const int rgbCameraId = 10;
    const int irCameraId = 1;

    const float confidenceThreshold = 0.2f;
    const float nmsThreshold = 0.4f;

    // Initialize TensorRT
    TensorRTContext ctx;
    if (!initTensorRT(ctx, enginePath)) {
        std::cerr << "Failed to initialize TensorRT engine" << std::endl;
        return -1;
    }

    // Open cameras
    cv::VideoCapture rgbCap(rgbCameraId);
    if (!rgbCap.isOpened()) {
        std::cerr << "Error: Could not open RGB camera " << rgbCameraId << std::endl;
        cleanupTensorRT(ctx);
        return -1;
    }

    cv::VideoCapture irCap(irCameraId);
    if (!irCap.isOpened()) {
        std::cerr << "Error: Could not open IR camera " << irCameraId << std::endl;
        rgbCap.release();
        cleanupTensorRT(ctx);
        return -1;
    }

    // Load calibration homography (thermal -> RGB)
    cv::Mat H;
    cv::FileStorage fs("calibration.yml", cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: Could not open calibration.yml" << std::endl;
        std::cerr << "Run the calibration tool first." << std::endl;
        rgbCap.release();
        irCap.release();
        cleanupTensorRT(ctx);
        return -1;
    }
    fs["H"] >> H;
    fs.release();
    std::cout << "Loaded calibration homography from calibration.yml" << std::endl;

    // Invert: H maps thermal->RGB, so H_inv maps RGB->thermal
    cv::Mat H_inv = H.inv();

    // IR display resolution (calibration was done at this size)
    const int irDisplayW = 640, irDisplayH = 480;

    auto prevTime = std::chrono::high_resolution_clock::now();

    std::cout << "=== Calibration RGB->IR Demo ===" << std::endl;
    std::cout << "Detecting ControlledFire (candle) and mapping box to IR view" << std::endl;
    std::cout << "Press 'q' to quit" << std::endl;

    while (true) {
        // Read RGB frame
        cv::Mat rgbFrame;
        if (!rgbCap.read(rgbFrame)) {
            std::cerr << "Error: Could not read RGB frame" << std::endl;
            break;
        }

        // Read and process IR frame
        cv::Mat irRawFrame;
        if (!irCap.read(irRawFrame)) {
            std::cerr << "Error: Could not read IR frame" << std::endl;
            break;
        }

        // Create IR heatmap display
        cv::Mat irTempFrame = convertToTemperature(irRawFrame);
        cv::Mat irDisplay;
        cv::normalize(irTempFrame, irDisplay, 0, 255, cv::NORM_MINMAX);
        irDisplay.convertTo(irDisplay, CV_8U);
        cv::applyColorMap(irDisplay, irDisplay, cv::COLORMAP_INFERNO);
        cv::resize(irDisplay, irDisplay, cv::Size(irDisplayW, irDisplayH));

        // Run YOLO inference
        DetectionResult result = processFrame(rgbFrame, ctx, confidenceThreshold, nmsThreshold);

        // Draw all detections on RGB
        cv::Mat rgbDisplay = rgbFrame.clone();
        drawDetections(rgbDisplay, result, ctx.class_names);

        // For ControlledFire detections, map bounding box to IR view
        for (int idx : result.indices) {
            if (result.class_ids[idx] != CONTROLLED_FIRE_ID) continue;

            cv::Rect box = result.boxes[idx];
            float conf = result.confidences[idx];

            // Get the 4 corners of the RGB bounding box
            std::vector<cv::Point2f> rgbCorners = {
                {(float)box.x, (float)box.y},
                {(float)(box.x + box.width), (float)box.y},
                {(float)(box.x + box.width), (float)(box.y + box.height)},
                {(float)box.x, (float)(box.y + box.height)}
            };

            // Map RGB corners -> IR display coordinates via inverse homography
            // H maps thermal_display(640x480) -> RGB, so H_inv maps RGB -> thermal_display(640x480)
            std::vector<cv::Point2f> irCorners;
            cv::perspectiveTransform(rgbCorners, irCorners, H_inv);

            // Draw the mapped bounding box on IR display
            for (int i = 0; i < 4; i++) {
                cv::line(irDisplay,
                         cv::Point((int)irCorners[i].x, (int)irCorners[i].y),
                         cv::Point((int)irCorners[(i + 1) % 4].x, (int)irCorners[(i + 1) % 4].y),
                         cv::Scalar(0, 255, 0), 2);
            }

            // Label on IR display
            std::string label = "ControlledFire " + std::to_string(conf).substr(0, 4);
            cv::putText(irDisplay, label,
                        cv::Point((int)irCorners[0].x, (int)irCorners[0].y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        // Calculate and draw FPS
        auto currTime = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(currTime - prevTime).count();
        prevTime = currTime;
        double fps = 1000.0 / ms;
        std::string fpsText = "FPS: " + std::to_string((int)fps);
        cv::putText(rgbDisplay, fpsText, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(irDisplay, fpsText, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // Display both views
        cv::imshow("RGB - YOLO Detection", rgbDisplay);
        cv::imshow("IR - Mapped BBox", irDisplay);

        if (cv::waitKey(1) == 'q') break;
    }

    // Cleanup
    rgbCap.release();
    irCap.release();
    cv::destroyAllWindows();
    cleanupTensorRT(ctx);

    return 0;
}

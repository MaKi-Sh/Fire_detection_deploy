#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

// States
enum State {
    LIVE,       // Normal camera feed
    PAUSED,     // Frozen, waiting for clicks
    CONFIRM     // Waiting for confirm/redo
};

// Global variables
State currentState = LIVE;
std::vector<cv::Point2f> rgbPoints;
std::vector<cv::Point2f> thermalPoints;
cv::Point2f tempRgbPoint;
cv::Point2f tempThermalPoint;
bool rgbClicked = false;
bool thermalClicked = false;

cv::Mat rgbFrozen, thermalFrozen;
cv::Mat rgbLive, thermalLive;

// Draw button on image
void drawButton(cv::Mat& img, std::string text, cv::Rect rect, cv::Scalar color) {
    cv::rectangle(img, rect, color, -1);
    cv::putText(img, text, {rect.x + 10, rect.y + 25}, 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);
}

// Check if click is inside button
bool clickedButton(int x, int y, cv::Rect rect) {
    return x >= rect.x && x <= rect.x + rect.width &&
           y >= rect.y && y <= rect.y + rect.height;
}

// Button positions
cv::Rect pauseBtn(10, 10, 100, 40);
cv::Rect confirmBtn(10, 10, 100, 40);
cv::Rect redoBtn(120, 10, 80, 40);
cv::Rect finishBtn(210, 10, 100, 40);

void rgbClick(int event, int x, int y, int flags, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) return;
    
    if (currentState == LIVE) {
        if (clickedButton(x, y, pauseBtn)) {
            // Freeze frames
            rgbFrozen = rgbLive.clone();
            thermalFrozen = thermalLive.clone();
            currentState = PAUSED;
            rgbClicked = false;
            thermalClicked = false;
            std::cout << "Paused - click calibration point on BOTH images" << std::endl;
        }
    }
    else if (currentState == PAUSED) {
        if (!clickedButton(x, y, pauseBtn)) {
            // Clicked a calibration point
            tempRgbPoint = {(float)x, (float)y};
            rgbClicked = true;
            std::cout << "RGB point: (" << x << ", " << y << ")" << std::endl;
            
            if (rgbClicked && thermalClicked) {
                currentState = CONFIRM;
            }
        }
    }
    else if (currentState == CONFIRM) {
        if (clickedButton(x, y, confirmBtn)) {
            // Save points
            rgbPoints.push_back(tempRgbPoint);
            thermalPoints.push_back(tempThermalPoint);
            std::cout << "Point " << rgbPoints.size() << " saved!" << std::endl;
            currentState = LIVE;
        }
        else if (clickedButton(x, y, redoBtn)) {
            // Redo
            currentState = PAUSED;
            rgbClicked = false;
            thermalClicked = false;
            std::cout << "Redo - click points again" << std::endl;
        }
        else if (clickedButton(x, y, finishBtn) && rgbPoints.size() >= 4) {
            // Calculate and save homography
            cv::Mat H = cv::findHomography(rgbPoints, thermalPoints);
            cv::FileStorage fs("calibration.yml", cv::FileStorage::WRITE);
            fs << "H" << H;
            fs.release();
            std::cout << "Calibration saved! " << rgbPoints.size() << " points used." << std::endl;
            exit(0);
        }
    }
}

void thermalClick(int event, int x, int y, int flags, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) return;
    
    if (currentState == PAUSED) {
        tempThermalPoint = {(float)x, (float)y};
        thermalClicked = true;
        std::cout << "Thermal point: (" << x << ", " << y << ")" << std::endl;
        
        if (rgbClicked && thermalClicked) {
            currentState = CONFIRM;
        }
    }
}

int main() {
    cv::VideoCapture rgb(0);
    cv::VideoCapture thermal("/dev/video1", cv::CAP_V4L2);
    thermal.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y','1','6',' '));
    thermal.set(cv::CAP_PROP_FRAME_WIDTH, 160);
    thermal.set(cv::CAP_PROP_FRAME_HEIGHT, 120);
    
    cv::namedWindow("RGB");
    cv::namedWindow("Thermal");
    cv::setMouseCallback("RGB", rgbClick);
    cv::setMouseCallback("Thermal", thermalClick);
    
    std::cout << "=== Calibration Tool ===" << std::endl;
    std::cout << "1. Click PAUSE" << std::endl;
    std::cout << "2. Click same point on BOTH windows" << std::endl;
    std::cout << "3. Click CONFIRM (or REDO)" << std::endl;
    std::cout << "4. Repeat 4+ times, then FINISH" << std::endl;
    
    while (true) {
        // Always grab live frames
        rgb >> rgbLive;
        cv::Mat thermalRaw;
        thermal >> thermalRaw;
        
        // Convert thermal for display
        cv::Mat thermalDisplay;
        cv::normalize(thermalRaw, thermalDisplay, 0, 255, cv::NORM_MINMAX);
        thermalDisplay.convertTo(thermalDisplay, CV_8U);
        cv::applyColorMap(thermalDisplay, thermalDisplay, cv::COLORMAP_INFERNO);
        cv::resize(thermalDisplay, thermalLive, cv::Size(640, 480));
        
        // Choose what to display
        cv::Mat rgbShow = (currentState == LIVE) ? rgbLive.clone() : rgbFrozen.clone();
        cv::Mat thermalShow = (currentState == LIVE) ? thermalLive.clone() : thermalFrozen.clone();
        
        // Draw UI based on state
        if (currentState == LIVE) {
            drawButton(rgbShow, "PAUSE", pauseBtn, cv::Scalar(0, 100, 255));
        }
        else if (currentState == PAUSED) {
            cv::putText(rgbShow, "Click calibration point", {10, 30}, 
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,255), 2);
            
            // Show clicked points
            if (rgbClicked) {
                cv::circle(rgbShow, tempRgbPoint, 8, cv::Scalar(0,255,0), 2);
            }
            if (thermalClicked) {
                cv::circle(thermalShow, tempThermalPoint, 8, cv::Scalar(0,255,0), 2);
            }
        }
        else if (currentState == CONFIRM) {
            drawButton(rgbShow, "CONFIRM", confirmBtn, cv::Scalar(0,200,0));
            drawButton(rgbShow, "REDO", redoBtn, cv::Scalar(0,0,200));
            
            if (rgbPoints.size() >= 3) {  // Show finish after 4th point will be added
                drawButton(rgbShow, "FINISH", finishBtn, cv::Scalar(200,0,0));
            }
            
            // Show clicked points
            cv::circle(rgbShow, tempRgbPoint, 8, cv::Scalar(0,255,0), 2);
            cv::circle(thermalShow, tempThermalPoint, 8, cv::Scalar(0,255,0), 2);
        }
        
        // Draw saved points
        for (int i = 0; i < rgbPoints.size(); i++) {
            cv::circle(rgbShow, rgbPoints[i], 5, cv::Scalar(255,0,0), -1);
            cv::putText(rgbShow, std::to_string(i+1), rgbPoints[i], 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);
        }
        
        // Show point count
        std::string countText = "Points: " + std::to_string(rgbPoints.size()) + "/4+";
        cv::putText(rgbShow, countText, {rgbShow.cols - 120, 30}, 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);
        
        cv::imshow("RGB", rgbShow);
        cv::imshow("Thermal", thermalShow);
        
        if (cv::waitKey(1) == 'q') break;
    }
    
    return 0;
}

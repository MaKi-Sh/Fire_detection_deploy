#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cstdint>

// Structure to hold individual hot pixel data
struct Pixel {
	int x, y;
	float temp;
};

// Structure to hold fire detection results from IR camera
struct FireResult {
	bool detected = false;
	float maxTemp = 0.0f;
	int largestClusterSize = 0;
	cv::Point hotspot = cv::Point(-1, -1);
};

// Structure combining IR detection with cluster data
struct IRDetection {
	FireResult info;
	std::vector<std::vector<Pixel>> clusters;
};

// Convert raw IR frame to float for processing.
// Camera outputs 8-bit intensity (0-255), not radiometric temperature.
// Values are relative: higher = hotter, but not calibrated to degrees.
cv::Mat convertToTemperature(const cv::Mat& rawFrame) {
	cv::Mat tempFrame;
	// Handle both 8-bit and 16-bit input
	if (rawFrame.type() == CV_8UC1) {
		rawFrame.convertTo(tempFrame, CV_32F);
	} else if (rawFrame.type() == CV_8UC3) {
		cv::Mat gray;
		cv::cvtColor(rawFrame, gray, cv::COLOR_BGR2GRAY);
		gray.convertTo(tempFrame, CV_32F);
	} else if (rawFrame.type() == CV_16UC1) {
		rawFrame.convertTo(tempFrame, CV_32F);
	} else {
		rawFrame.convertTo(tempFrame, CV_32F);
	}
	return tempFrame;
}

// Detect fire hotspots from IR frame using relative thresholding.
// tempThreshold is a stddev multiplier: pixels above (mean + tempThreshold * stddev)
// are considered "hot". A value of 3.0-5.0 works well for fire detection.
IRDetection fireDetect(const cv::Mat& tempFrame, float tempThreshold, int minClusterSize) {
	IRDetection detection;

	// Find max value and location
	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(tempFrame, &minVal, &maxVal, &minLoc, &maxLoc);
	detection.info.maxTemp = static_cast<float>(maxVal);
	detection.info.hotspot = maxLoc;

	// Compute dynamic threshold: mean + k * stddev
	cv::Scalar meanVal, stddevVal;
	cv::meanStdDev(tempFrame, meanVal, stddevVal);
	float dynamicThreshold = static_cast<float>(meanVal[0] + tempThreshold * stddevVal[0]);

	// Clamp threshold to valid range
	if (dynamicThreshold > maxVal) dynamicThreshold = static_cast<float>(maxVal);

	// Create binary mask of hot pixels
	cv::Mat mask;
	cv::threshold(tempFrame, mask, dynamicThreshold, 255, cv::THRESH_BINARY);

	cv::Mat mask8u;
	mask.convertTo(mask8u, CV_8U);

	// Find connected components (hot pixel clusters)
	cv::Mat labels, stats, centroids;
	int numLabels = cv::connectedComponentsWithStats(mask8u, labels, stats, centroids);

	// Check if any cluster is large enough to be considered fire
	for (int i = 1; i < numLabels; i++) {
		int area = stats.at<int>(i, cv::CC_STAT_AREA);
		if (area >= minClusterSize) {
			detection.info.detected = true;
			if (area > detection.info.largestClusterSize) {
				detection.info.largestClusterSize = area;
			}
		}
	}

	// Build pixel clusters (only if we have labels beyond background)
	if (numLabels > 1) {
		detection.clusters.resize(numLabels - 1);
		for (int y = 0; y < labels.rows; y++) {
			for (int x = 0; x < labels.cols; x++) {
				int label = labels.at<int>(y, x);
				if (label > 0) {
					Pixel p;
					p.x = x;
					p.y = y;
					p.temp = tempFrame.at<float>(y, x);
					detection.clusters[label - 1].push_back(p);
				}
			}
		}
	}

	return detection;
}

// Check if a point is inside a bounding box
bool isPointInBox(int px, int py, const cv::Rect& box) {
	return px >= box.x && px < box.x + box.width &&
	       py >= box.y && py < box.y + box.height;
}

// Scale pixel coordinates from IR frame to RGB frame dimensions
cv::Point scalePixelCoords(const Pixel& pix, const cv::Size& irSize, const cv::Size& rgbSize) {
	float scaleX = static_cast<float>(rgbSize.width) / irSize.width;
	float scaleY = static_cast<float>(rgbSize.height) / irSize.height;
	return cv::Point(static_cast<int>(pix.x * scaleX), static_cast<int>(pix.y * scaleY));
}

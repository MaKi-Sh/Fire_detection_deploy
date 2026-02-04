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

// Convert raw IR frame (uint16_t) to temperature in Celsius
cv::Mat convertToTemperature(const cv::Mat& rawFrame) {
	cv::Mat tempFrame(rawFrame.size(), CV_32F);
	for (int i = 0; i < rawFrame.rows; i++) {
		for (int j = 0; j < rawFrame.cols; j++) {
			uint16_t rawVal = rawFrame.at<uint16_t>(i, j);
			tempFrame.at<float>(i, j) = (static_cast<float>(rawVal) / 100.0f) - 273.15f;
		}
	}
	return tempFrame;
}

// Detect fire from IR temperature frame
IRDetection fireDetect(const cv::Mat& tempFrame, float tempThreshold, int minClusterSize) {
	IRDetection detection;

	// Find max temperature and location
	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(tempFrame, &minVal, &maxVal, &minLoc, &maxLoc);
	detection.info.maxTemp = static_cast<float>(maxVal);
	detection.info.hotspot = maxLoc;

	// Create binary mask of hot pixels
	cv::Mat mask;
	cv::threshold(tempFrame, mask, tempThreshold, 255, cv::THRESH_BINARY);

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

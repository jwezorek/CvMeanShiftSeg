#pragma once

#include <opencv2/core/core.hpp>

class MeanShiftSegmentation : public cv::Algorithm {
public:

	virtual void processImage(cv::InputArray src, cv::OutputArray segmented) = 0;
	virtual void processImage(cv::InputArray src, cv::OutputArray segmented, cv::OutputArray labelMap) = 0;

	virtual void setSigmaS(int val) = 0;
	virtual int getSigmaS() const = 0;

	virtual void setSigmaR(float val) = 0;
	virtual float getSigmaR() const = 0;

	virtual void setMinSize(int min_size) = 0;
	virtual int getMinSize() = 0;

	virtual void setOptimized(bool val) = 0;
	virtual bool getOptimized() const = 0;

	virtual void setConnectivity(int n) = 0;
	virtual int getConnectivity() const = 0;

};

cv::Ptr<MeanShiftSegmentation> createMeanShiftSegmentation(int sigmaS, float sigmaR, int minSize, int connectivity, bool optimized);
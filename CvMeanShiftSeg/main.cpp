#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "MeanShiftSegmentation.hpp"

int main(int argc, char **argv)
{
	if (argc != 2)
		return -1;

	cv::Mat img = cv::imread(argv[1]);

	cv::Mat output;
	cv::Mat labels;
	auto mss = createMeanShiftSegmentation(8, 5.5f, 20, 4, true);
	mss->processImage(img, output, labels);

	cv::imshow("segmented output", output);
	cv::imwrite("c:\\work\\yoshi-segmented.png", output);

	cv::waitKey();
	return 0;
}
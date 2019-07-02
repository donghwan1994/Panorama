#include "Moravec.h"
#include "HOG_Descriptor.h"
#include "Euclidean.h"
#include "Panorama.h"

using namespace IPCVL;

int main()
{
	cv::Mat src1 = imread("¼Ò³à½Ã´ë_1.jpg", cv::IMREAD_GRAYSCALE);
	int threshold = 20000;
	Moravec moravec1(src1, threshold);
	moravec1.CreatConfidence(src1);

	cv::Mat src2 = imread("¼Ò³à½Ã´ë_2.jpg", cv::IMREAD_GRAYSCALE);
	Moravec moravec2 (src2, threshold);
	moravec2.CreatConfidence(src2);

	cv::Mat dst1 = imread("¼Ò³à½Ã´ë_1.jpg", cv::IMREAD_COLOR);
	cv::Mat dst2 = imread("¼Ò³à½Ã´ë_2.jpg", cv::IMREAD_COLOR);
	
	vector<pair<int, int>> keyPoint1, keyPoint2;

	keyPoint1 = moravec1.getKeyPoint();
	keyPoint2 = moravec2.getKeyPoint();

	HOG_Descriptor HOGDescriptor1(src1, keyPoint1, 16);
	HOG_Descriptor HOGDescriptor2(src2, keyPoint2, 16);

	HOGDescriptor1.MakeDescriptor();
	HOGDescriptor2.MakeDescriptor();

	vector<FeatureVector> featureVector1 = HOGDescriptor1.getFeatureVector();
	vector<FeatureVector> featureVector2 = HOGDescriptor2.getFeatureVector();

	Euclidean euclidean;

	euclidean.getMinimumDistanceKeyPoint(featureVector1, featureVector2, 1.);

	std::vector<std::pair<cv::Point, cv::Point>> similarKeyPoints;

	similarKeyPoints = euclidean.getSimilarKeypoints();

	Panorama panorama(dst1, dst2);

	cv::Mat T = panorama.MatchingByRANSAC(similarKeyPoints, 100, 50, 4.);

	cv::Mat dst3 = panorama.makePanoramaImage(T, similarKeyPoints);

	imshow("dst1", dst1);
	imshow("dst2", dst2);
	imshow("dst3", dst3);

	cv::waitKey();
	
	return 0;
}
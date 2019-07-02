#include "Moravec.h"
using namespace std;

namespace IPCVL
{
	Moravec::Moravec(const cv::Mat &src, const int &threshold)
	{
		confidence = cv::Mat::zeros(src.size(), CV_64FC1);

		this->threshold = threshold;
	}

	void Moravec::CreatConfidence(const cv::Mat &src)
	{
		for (int y = 2; y < src.rows - 2; ++y)
		{
			for (int x = 2; x < src.cols - 2; ++x)
			{
				processChannel(src, x, y);
			}
		}
	}

	void Moravec::processChannel(const cv::Mat &img, const int &x, const int &y)
	{
		std::vector<float> sum_of_squared_difference(4);

		for (int j = -1; j < 2; ++j)
		{
			for (int i = -1; i < 2; ++i)
			{
				sum_of_squared_difference[0] += pow((img.at<uchar>(y + j, x + i + 1) - img.at<uchar>(y + j, x + i)), 2);
				sum_of_squared_difference[1] += pow((img.at<uchar>(y + j, x + i - 1) - img.at<uchar>(y + j, x + i)), 2);
				sum_of_squared_difference[2] += pow((img.at<uchar>(y + j + 1, x + i) - img.at<uchar>(y + j, x + i)), 2);
				sum_of_squared_difference[3] += pow((img.at<uchar>(y + j - 1, x + i) - img.at<uchar>(y + j, x + i)), 2);
			}
		}

		confidence.at<double>(y, x) = min(sum_of_squared_difference[3],
				min(sum_of_squared_difference[2],
				min(sum_of_squared_difference[1],
					sum_of_squared_difference[0])));

		if (confidence.at<double>(y, x) >= threshold)
			StoreFeaturePossiblity(x, y);
	}

	void Moravec::StoreFeaturePossiblity(const int &x, const int &y)
	{
		if (confidence.at<double>(y, x) >= threshold)
			keyPoint.push_back(pair<int, int>(x, y));
	}

	void Moravec::DrawFeature(const cv::Mat &dst)
	{
		vector<pair<int, int>>::iterator iter;

		for (iter = begin(keyPoint); iter != end(keyPoint); ++iter)
			circle(dst, cv::Point(iter->first, iter->second), 3, cv::Scalar(255, 0, 0));

	}

	vector<pair<int, int>> Moravec::getKeyPoint()
	{
		return keyPoint;
	}
}

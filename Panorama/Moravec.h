#pragma once
#include "opencv2\opencv.hpp"
#include <utility>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace cv;
using namespace std;

namespace IPCVL
{
	class Moravec 
	{
	private:
		cv::Mat confidence;
		vector<pair<int, int>> keyPoint;
		int threshold;

	public:
		Moravec(const cv::Mat &src, const int &threshold);
		void CreatConfidence(const cv::Mat &src);
		void DrawFeature(const cv::Mat& dst);
		void processChannel(const cv::Mat &img, const int &x, const int &y);
		void StoreFeaturePossiblity(const int &x, const int &y);
		vector<pair<int, int>> getKeyPoint();
	};
}
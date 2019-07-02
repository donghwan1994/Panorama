#pragma once
#include "FeatureVector.h"
#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
using namespace cv;
using namespace std;

namespace IPCVL
{		
	class Euclidean
	{
	private:
		vector<pair<cv::Point, cv::Point>> similarKeypoints;

	public:
		void getMinimumDistanceKeyPoint(vector<FeatureVector> &featurevector1,
			vector<FeatureVector> &featurevector2, const double &threshold);

		double getMinimumDistance(const vector<vector<double>> &vector1,
			const vector<vector<double>> &vector2, double &distance);

		vector<pair<cv::Point, cv::Point>> getSimilarKeypoints();

		void DrawSimilarKeyPoints(Mat &img1, Mat &img2);
	};
}
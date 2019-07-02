#pragma once
#include "opencv2\opencv.hpp"
#include <utility>
#include <vector>
#include <tuple>
#include <math.h>
using namespace std;

namespace IPCVL
{
	class Panorama
	{
	private:
		//vector<pair<cv::Point, cv::Point>> similarKeypoints;
		cv::Mat &img1;
		cv::Mat &img2;		

		void makeInput_Matrix(const vector<pair<cv::Point, cv::Point>> &similarKeypoints, 
			cv::Mat &input, vector<pair<cv::Point, cv::Point>>::iterator &iter);
		void makeOutput_Matrix(cv::Mat &input, vector<pair<cv::Point, cv::Point>>::iterator &iter);
		void makeTransformationMatrix(const cv::Mat &input, cv::Mat &output);

	public:
		Panorama(cv::Mat &img1, cv::Mat &img2);
		cv::Mat getTransformationMatrixByleastSquareMethod(vector<pair<cv::Point, cv::Point>> &similarKeypoints);
		cv::Mat MatchingByRANSAC(const vector<pair<cv::Point, cv::Point>> &similarKeypoints,
			const int &k, const int &d, const int &fiterror);
		int getHeight();
		int getWidth(const vector<pair<cv::Point, cv::Point>> &similarKeypoints);
		cv::Mat makePanoramaImage(const cv::Mat &TransformationMatrix, 
			const vector<pair<cv::Point, cv::Point>> &similarKeypoints);
		cv::Mat getTransXY(const int &x, const int &y, const cv::Mat &TransformationMatrix);
	};
}
#include "Panorama.h"
#include "RandomNumberGenerator.h"
#include <string.h>

IPCVL::Panorama::Panorama(cv::Mat &img1, cv::Mat &img2)
	:img1{ img1 }, img2{ img2 }
{}

void IPCVL::Panorama::makeInput_Matrix(const vector<pair<cv::Point, cv::Point>> &similarKeypoints,
	cv::Mat &input, vector<pair<cv::Point, cv::Point>>::iterator &iter)
{
	input.at<double>(0, 0) += pow(iter->second.y, 2);
	input.at<double>(0, 1) += iter->second.y * iter->second.x;
	input.at<double>(0, 2) += iter->second.y;
	input.at<double>(1, 1) += pow(iter->second.x, 2);
	input.at<double>(1, 2) += iter->second.x;
	input.at<double>(2, 2) += 1;

	if (iter == end(similarKeypoints) - 1)
	{
		input.at<double>(1, 0) = input.at<double>(0, 1);
		input.at<double>(2, 0) = input.at<double>(0, 2);
		input.at<double>(2, 1) = input.at<double>(1, 2);

		for (int j = 0; j < 3; ++j)
			for (int i = 0; i < 3; ++i)
				input.at<double>(j + 3, i + 3) = input.at<double>(j, i);
	}
}

void IPCVL::Panorama::makeOutput_Matrix(cv::Mat &input,
	vector<pair<cv::Point, cv::Point>>::iterator &iter)
{
	input.at<double>(0, 0) += iter->first.y * iter->second.y;
	input.at<double>(1, 0) += iter->second.x * iter->first.y;
	input.at<double>(2, 0) += iter->first.y;
	input.at<double>(3, 0) += iter->second.y * iter->first.x;
	input.at<double>(4, 0) += iter->first.x * iter->second.x;
	input.at<double>(5, 0) += iter->first.x;
}

void IPCVL::Panorama::makeTransformationMatrix(const cv::Mat &input, cv::Mat &output)
{
	output.at<double>(0, 0) = input.at<double>(0, 0);
	output.at<double>(0, 1) = input.at<double>(3, 0);
	output.at<double>(1, 0) = input.at<double>(1, 0);
	output.at<double>(1, 1) = input.at<double>(4, 0);
	output.at<double>(2, 0) = input.at<double>(2, 0);
	output.at<double>(2, 1) = input.at<double>(5, 0);
	output.at<double>(2, 2) = 1;
}

int IPCVL::Panorama::getHeight()
{
	return img2.rows;
}

int IPCVL::Panorama::getWidth(const vector<pair<cv::Point, cv::Point>> &similarKeypoints)
{
	return similarKeypoints[0].first.x
		+ img2.cols - similarKeypoints[0].second.x;
}

cv::Mat IPCVL::Panorama::getTransXY(const int &x, const int &y, const cv::Mat &TransformationMatrix)
{
	double array[1][3]{ {y, x, 1} };
	cv::Mat homogeneousCoordinate(1, 3, CV_64FC1, array);
	cv::Mat point_dst;
	point_dst = homogeneousCoordinate * TransformationMatrix;

	return point_dst;
}

cv::Mat IPCVL::Panorama::getTransformationMatrixByleastSquareMethod(vector<pair<cv::Point, cv::Point>> &similarKeypoints)
{
	cv::Mat input_Matrix = cv::Mat::zeros(6, 6, CV_64FC1);
	cv::Mat operand_Matrix = cv::Mat::zeros(6, 1, CV_64FC1);
	cv::Mat output_Matrix = cv::Mat::zeros(6, 1, CV_64FC1);

	vector<pair<cv::Point, cv::Point>>::iterator iter;

	for (iter = begin(similarKeypoints); iter != end(similarKeypoints); ++iter)
	{
		makeInput_Matrix(similarKeypoints, input_Matrix, iter);
		makeOutput_Matrix(output_Matrix, iter);
	}

	operand_Matrix = input_Matrix.inv() * output_Matrix;

	cv::Mat transformationMatrix = cv::Mat::zeros(3, 3, CV_64FC1);

	makeTransformationMatrix(operand_Matrix, transformationMatrix);

	return transformationMatrix;
}

cv::Mat IPCVL::Panorama::MatchingByRANSAC(const vector<pair<cv::Point, cv::Point>> &similarKeypoints,
	const int &k, const int &d, const int &fiterror)
{
	cv::Mat output;
	int minError = numeric_limits<int>::max();
	for (int i = 0; i < k; ++i)
	{
		vector<pair<cv::Point, cv::Point>> inlier;
		vector<pair<cv::Point, cv::Point>> randomSimilarKeypoint;
		vector<int> threeRandomNumber;
		for (int j = 0; j < 3; ++j)
		{
			int randomNumber = UTIL::getRandomNumber(0, similarKeypoints.size());
			randomSimilarKeypoint.push_back(similarKeypoints[randomNumber]);
			inlier.push_back(similarKeypoints[randomNumber]);
			threeRandomNumber.push_back(randomNumber);
		}

		cv::Mat TransformationMatrix = getTransformationMatrixByleastSquareMethod(randomSimilarKeypoint);

		for (int j = 0; j < similarKeypoints.size(); ++j)
		{
			if (j == threeRandomNumber[0] || j == threeRandomNumber[1]
				|| j == threeRandomNumber[2])
				continue;

			cv::Mat point_dst = getTransXY(similarKeypoints[j].second.x,
				similarKeypoints[j].second.y, TransformationMatrix);

			int y_trans = round(point_dst.at<double>(0, 0));
			int x_trans = round(point_dst.at<double>(0, 1));

			int error = (pow(similarKeypoints[j].first.y - y_trans, 2)
				+ pow(similarKeypoints[j].first.x - x_trans, 2));

			if (error <= fiterror)
			{
				inlier.push_back(similarKeypoints[j]);
				output = TransformationMatrix;
			}
		}

		if (inlier.size() >= d)
		{
			TransformationMatrix = getTransformationMatrixByleastSquareMethod(inlier);

			int error = 0;
			
			for (int k = 0; k < inlier.size(); ++k)
			{
				cv::Mat point_dst = getTransXY(inlier[k].second.x,
					inlier[k].second.y, TransformationMatrix);

				int y_trans = round(point_dst.at<double>(0, 0));
				int x_trans = round(point_dst.at<double>(0, 1));
				error += pow(inlier[k].first.y - y_trans, 2)
					+ pow(inlier[k].first.x - x_trans, 2);
			}

			if (error <= minError)
			{
				minError = error;
				output = TransformationMatrix;
			}
		}
	}

	return output;
}

cv::Mat IPCVL::Panorama::makePanoramaImage(const cv::Mat &TransformationMatrix,
	const vector<pair<cv::Point, cv::Point>> &similarKeypoints)
{
	int height = getHeight();
	int width = getWidth(similarKeypoints);

	cv::Mat outputMat(cv::Size(width, height), CV_8UC3);

	for (int y = 0; y < img1.rows; ++y)
	{
		for (int x = 0; x < img1.cols; ++x)
		{
			outputMat.at<cv::Vec3b>(y, x)[0] = img1.at<cv::Vec3b>(y, x)[0];
			outputMat.at<cv::Vec3b>(y, x)[1] = img1.at<cv::Vec3b>(y, x)[1];
			outputMat.at<cv::Vec3b>(y, x)[2] = img1.at<cv::Vec3b>(y, x)[2];
		}
	}

	for (int y = 0; y < img2.rows; ++y)
	{
		for (int x = 0; x < img2.cols; ++x)
		{
			cv::Mat point_dst = getTransXY(x, y, TransformationMatrix);

			int y_trans = round(point_dst.at<double>(0, 0));
			int x_trans = round(point_dst.at<double>(0, 1));

			if (y_trans < outputMat.rows && y_trans >= 0
				&& x_trans < outputMat.cols && x_trans >= 0)
			{
				outputMat.at<cv::Vec3b>(y_trans, x_trans)[0] = img2.at<cv::Vec3b>(y, x)[0];
				outputMat.at<cv::Vec3b>(y_trans, x_trans)[1] = img2.at<cv::Vec3b>(y, x)[1];
				outputMat.at<cv::Vec3b>(y_trans, x_trans)[2] = img2.at<cv::Vec3b>(y, x)[2];
			}

		}
	}

	return outputMat;
}
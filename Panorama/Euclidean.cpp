#include "Euclidean.h"

void IPCVL::Euclidean::getMinimumDistanceKeyPoint(vector<FeatureVector> &featurevector1,
	vector<FeatureVector> &featurevector2, const double &threshold)
{
	vector<FeatureVector>::iterator iter1, iter2;

	for (iter1 = begin(featurevector1); iter1 != end(featurevector1); ++iter1)
	{
		double minDistance = numeric_limits<double>::max();
		bool isSimilarKeyPoints = false;
		int min_y = 0, min_x = 0;
		for (iter2 = begin(featurevector2); iter2 != end(featurevector2); ++iter2)
		{
			double temp = minDistance;
			minDistance = getMinimumDistance(iter1->vector, iter2->vector, minDistance);

			if (minDistance != temp && minDistance < threshold)
			{
				min_x = iter2->x;
				min_y = iter2->y;
				isSimilarKeyPoints = true;
			}
		}

		if(isSimilarKeyPoints == true)
			similarKeypoints.push_back(pair<Point, Point>
				(Point(iter1->x, iter1->y), Point(min_x, min_y)));
	}
}

double IPCVL::Euclidean::getMinimumDistance(const vector<vector<double>> &vector1,
	const vector<vector<double>> &vector2, double &distance)
{
	double sum = 0.;

	for (int i = 0; i < vector1.size(); ++i)
		for (int j = 0; j < vector1[0].size(); ++j)
			sum += pow(vector1[i][j] - vector2[i][j], 2);

	sum = sqrt(sum);

	return min(distance, sum);
}

vector<pair<cv::Point, cv::Point>> IPCVL::Euclidean::getSimilarKeypoints()
{
	return similarKeypoints;
}

void IPCVL::Euclidean::DrawSimilarKeyPoints(Mat &img1, Mat &img2)
{
	vector<pair<Point, Point>>::iterator iter;

	for (iter = begin(similarKeypoints); iter != end(similarKeypoints); ++iter)
	{
		circle(img1, Point(iter->first.x, iter->first.y), 3, Scalar(0, 0, 255));
		circle(img2, Point(iter->second.x, iter->second.y), 3, Scalar(0, 0, 255));
	}		
}
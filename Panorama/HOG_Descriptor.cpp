#include "HOG_Descriptor.h"


IPCVL::HOG_Descriptor::HOG_Descriptor(const Mat &src, vector<pair<int, int>> &input, const int &windowSize=16)
{
	src.copyTo(img);
	this->windowSize = windowSize;
	this->blockSize = windowSize / 4;

	vector<std::pair<int, int>>::iterator iter;

	for (iter = begin(input); iter != end(input); iter++)
		keyPoint.push_back(pair<int, int>(iter->first, iter->second));
}

void IPCVL::HOG_Descriptor::getHistogram(vector<double> &histogram,
	const vector<double> &Orientation, const vector<double> &weight)
{
	for (int i = 0; i < Orientation.size(); ++i)
	{
		int bin = round(Orientation[i] / (360 >> 3));
		if (bin >= 8)
			bin -= 8;
		if (bin < 0)
			bin += 8;
		histogram[bin] += (1 * weight[i]);
	}

	double sum_of_hist = 0;

	for (auto &e : histogram)
		sum_of_hist += e;

	for (int i = 0; i < histogram.size(); ++i)
		histogram[i] = histogram[i] / sum_of_hist;
}
		
void IPCVL::HOG_Descriptor::MakeDescriptor()
{
	const int SIFT_ORI_HIST_BINS = 8;
	vector<std::pair<int, int>>::iterator iter;
	vector<vector<double>> feature_vector_oneKeypoint;
	for (iter = begin(keyPoint); iter != end(keyPoint); iter++)
	{
		cv::Point beginPoint = cv::Point(iter->first - windowSize/2, iter->second - windowSize/2);
		cv::Point endPoint = cv::Point(iter->first + windowSize/2, iter->second + windowSize/2);
		for (int y1 = beginPoint.y; y1 < endPoint.y; y1 += blockSize)
		{
			for (int x1 = beginPoint.x; x1 < endPoint.x; x1 += blockSize)
			{
				vector<double> orient;
				vector<double> mag;

				for (int y2 = y1; y2 < y1 + blockSize; ++y2)
				{
					for (int x2 = x1; x2 < x1 + blockSize; ++x2)
					{
						if (y2 > 0 && y2 < img.rows - 1 && 
							x2 > 0 && x2 < img.cols - 1)
						{
							int dx = -img.at<uchar>(y2, x2 - 1) 
								+ img.at<uchar>(y2, x2 + 1);
							int dy = -img.at<uchar>(y2 - 1, x2) 
								+ img.at<uchar>(y2 + 1, x2);

							cv::Mat Mag(cv::Size(1, 1), CV_64FC1);
							cv::Mat Angle(cv::Size(1, 1), CV_64FC1);

							cv::cartToPolar(dx, dy, Mag, Angle, true);

							orient.push_back(Angle.at<double>(0));
							mag.push_back(Mag.at<double>(0));
						}								
					}
				}

				vector<double> HOG(SIFT_ORI_HIST_BINS, 0);

				getHistogram(HOG, orient, mag);
					
				feature_vector_oneKeypoint.push_back(HOG);
			}				
		}

		featureVector.push_back(IPCVL::FeatureVector(iter->first,
			iter->second, feature_vector_oneKeypoint));

		feature_vector_oneKeypoint.clear();
	}
}

vector<IPCVL::FeatureVector> IPCVL::HOG_Descriptor::getFeatureVector() 
{ 
	return featureVector; 
}
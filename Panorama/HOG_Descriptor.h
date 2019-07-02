#pragma once
#include "FeatureVector.h"
#include "Moravec.h"
#include <vector>
#include <utility>
#include <algorithm>
#include "FeatureVector.h"
using namespace std;

namespace IPCVL
{
	class HOG_Descriptor
	{
	private:
		Mat img;
		vector<pair<int, int>> keyPoint;		
		vector<FeatureVector> featureVector;
		int windowSize;
		int blockSize;

	public:
		HOG_Descriptor(const Mat &src, vector<pair<int, int>> &input, const int &windowSize);
		vector<FeatureVector> getFeatureVector();
		void FindDominantOrientation();
		void MakeDescriptor();
		void getHistogram(vector<double> &histogram, const vector<double> &gradientOrientation,
			const vector<double> &dominantOrientation);
	};
}

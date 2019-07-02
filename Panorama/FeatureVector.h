#pragma once
#include "opencv2\opencv.hpp"

namespace IPCVL
{
	class FeatureVector
	{
	public:
		int y;
		int x;
		std::vector<std::vector<double>> vector;

		FeatureVector(int &x, int &y, std::vector<std::vector<double>> &featureVector)
			: x(x), y(y), vector(featureVector)
		{}
	};
}
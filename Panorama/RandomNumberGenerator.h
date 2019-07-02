#pragma once
#include <iostream>
#include <cstdlib>	// std::radn(), std::srand()
#include <ctime>	// std::time()
#include <random>

using namespace std;
namespace UTIL
{
	unsigned int PRNG() // Pseudo Random Number Generator
	{
		static unsigned int seed = 5523; // seed number

		seed = 8253729 * seed + 2396403;

		return seed % 32768;
	}

	int getRandomNumber(int min, int max)
	{
		static const double fraction = 1. / (RAND_MAX + 1.);

		return min + static_cast<int>((max - min + 1) * (std::rand() * fraction));
	}
}
#include "TrainSet.h"

namespace NeuralNetwork
{
	TrainSet::TrainSet(double* input, double* output)
	{
		_input = input;
		_output = output;
	}

	TrainSet::~TrainSet()
	{
	}

} // namespace NeuralNetwork
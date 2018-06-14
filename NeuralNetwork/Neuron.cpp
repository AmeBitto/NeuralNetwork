#include "Neuron.h"

namespace NeuralNetwork
{
	Neuron::Neuron(int inputCount, bool isBias)
	{
		_isBias = isBias;
		_memory = 1.0f;

		if (!_isBias)
		{
			_weightCount = inputCount;
			_weights = new double[_weightCount];
			_chachedWeights = new double[_weightCount];
			_memory = 0.0f;
		}
	}

	Neuron::~Neuron()
	{
		if (!_isBias)
		{
			delete[] _weights;
			delete[] _chachedWeights;
		}
	}

	void Neuron::setWeights(double* weights)
	{
		for (int w = 0; w < _weightCount; w++)
		{
			_weights[w] = weights[w];
		}
	}

	void Neuron::setWeight(int index, double weight)
	{
		_weights[index] = weight;
	}

	double Neuron::getOutput(double* input)
	{
		if (_isBias) return 1.0f;

		double result = 0;
		for (int w = 0; w < _weightCount; w++)
		{
			result += input[w] * _weights[w];
		}
		_memory = result;
		return result;
	}

	void Neuron::setChachedWeights(double* weights)
	{
		for (int w = 0; w < _weightCount; w++)
		{
			_chachedWeights[w] = weights[w];
		}
	}

	void Neuron::setChachedWeight(int index, double weight)
	{
		_chachedWeights[index] = weight;
	}

	// Speed up (~ x2)
	void Neuron::loadWeightsFromChache()
	{
		for (int w = 0; w < _weightCount; w++)
		{
			_weights[w] = _chachedWeights[w];
		}
	}

} // namespace NeuralNetwork
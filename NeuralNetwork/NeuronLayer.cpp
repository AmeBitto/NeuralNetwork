#include "NeuronLayer.h"
#include <random>
#include <time.h>

namespace NeuralNetwork
{
	NeuronLayer::NeuronLayer(int inputCount, int neuronCount, bool useBias)
	{
		_neuronCount = neuronCount;
		if (useBias) _neuronCount++;

		_neurons = new Neuron*[_neuronCount];
		for (int n = 0; n < _neuronCount - 1; n++)
		{
			_neurons[n] = new Neuron(inputCount);
		}

		_neurons[_neuronCount - 1] = new Neuron(inputCount, useBias);
	}

	NeuronLayer::~NeuronLayer()
	{
		for (int n = 0; n < _neuronCount; n++)
		{
			delete _neurons[n];
		}

		delete[] _neurons;
	}

	void NeuronLayer::setupWeights(double** weights)
	{
		int neuronCount = _neuronCount;
		if (_neurons[_neuronCount - 1]->isBias()) neuronCount--;

		for (int n = 0; n < neuronCount; n++)
		{
			_neurons[n]->setWeights(weights[n]);
		}
	}

	void NeuronLayer::randomWeights()
	{
		srand(time(NULL));

		int neuronCount = _neuronCount;
		if (_neurons[_neuronCount - 1]->isBias()) neuronCount--;

		for (int n = 0; n < neuronCount; n++)
		{
			for (int w = 0; w < _neurons[n]->getWeightCount(); w++)
			{
				double weight = (double)rand() / (double)RAND_MAX;
				if (rand() % 2 == 1) weight *= -1;
				_neurons[n]->setWeight(w, weight);
			}

		}
	}

	void NeuronLayer::loadWeightsFromChache()
	{
		for (int n = 0; n < _neuronCount; n++)
		{
			_neurons[n]->loadWeightsFromChache();
		}
	}

} // namespace NeuralNetwork
#include "NeuronSettings.h"

namespace NeuralNetwork
{
	NeuronSettings::NeuronSettings(double learningRate, int layerCount, int inputCount, bool inputUseBias)
	{
		_learningRate = learningRate;

		_layerCount = layerCount;
		_layerCounter = 0;
		_layers = new NeuronLayerSettings*[_layerCount];

		_inputCount = inputCount;
		_inputUseBias = inputUseBias;
	}

	NeuronSettings::~NeuronSettings()
	{
		delete[] _layers;
	}

	void NeuronSettings::addLayer(NeuronLayerSettings* layer)
	{
		if (_layerCounter >= _layerCount)
		{
			// Max layers
			return;
		}

		_layers[_layerCounter++] = layer;
	}

} // namespace NeuralNetwork
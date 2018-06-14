#include "NeuronLayerSettings.h"

namespace NeuralNetwork
{
	NeuronLayerSettings::NeuronLayerSettings(int neuronCount, bool useBias)
	{
		_neuronCount = neuronCount;
		_isBiasUsed = useBias;
	}

	NeuronLayerSettings::~NeuronLayerSettings()
	{
	}

	void NeuronLayerSettings::setupWeights(double** weights)
	{
		_weights = weights;
	}

} // namespace NeuralNetwork
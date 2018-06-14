#ifndef NEURON_LAYER_H
#define NEURON_LAYER_H

#include "Neuron.h"

namespace NeuralNetwork
{
	class NeuronLayer
	{
	public:
		NeuronLayer(int inputCount, int neuronCount, bool useBias = false);
		~NeuronLayer();

		inline bool isBiasUsed() const { return _neurons[_neuronCount - 1]->isBias(); }

		inline Neuron** getNeurons() const { return _neurons; }
		inline int getNeuronCount() const { return _neuronCount; }

		void setupWeights(double** weights);
		void randomWeights();
		void loadWeightsFromChache();

	private:
		Neuron** _neurons;
		int _neuronCount;
	}; // class NeuronLayer
} // namespace NeuralNetwork

#endif // NEURON_LAYER_H
#ifndef NEURON_LAYER_SETTINGS_H
#define NEURON_LAYER_SETTINGS_H

namespace NeuralNetwork
{
	class NeuronLayerSettings
	{
	public:
		NeuronLayerSettings(int neuronCount, bool useBias = false);
		~NeuronLayerSettings();

		inline int getNeuronCount() const { return _neuronCount; }
		inline bool isBiasUsed() const { return _isBiasUsed; }
		inline double** getWeights() const { return _weights; }

		void setupWeights(double** weights);

	private:
		int _neuronCount;
		bool _isBiasUsed;

		double** _weights;
	}; // class NeuronLayerSettings
} // namespace NeuralNetwork

#endif // NEURON_LAYER_SETTINGS_H
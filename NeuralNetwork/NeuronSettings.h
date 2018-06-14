#ifndef NEURON_SETTINGS_H
#define NEURON_SETTINGS_H

#include "NeuronLayerSettings.h"

namespace NeuralNetwork
{
	class NeuronSettings
	{
	public:
		NeuronSettings(double learningRate, int layerCount, int inputCount, bool inputUseBias = false);
		~NeuronSettings();

		inline double getLearningRate() const { return _learningRate; }

		inline NeuronLayerSettings** getLayers() const { return _layers; }
		inline int getLayerCount() const { return _layerCount; }
		void addLayer(NeuronLayerSettings* layer);

		inline int getInputCount() const { return _inputCount; }
		inline bool isInputBiasUsed() const { return _inputUseBias; }

	private:
		double _learningRate;
		int _inputCount;
		bool _inputUseBias;

		NeuronLayerSettings** _layers;
		int _layerCount;
		int _layerCounter;
	}; // class NeuronSettings
} // namespace NeuralNetwork

#endif // NEURON_SETTINGS_H
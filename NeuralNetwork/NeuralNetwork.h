#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <math.h>

#include "NeuronLayer.h"
#include "NeuronSettings.h"
#include "TrainSet.h"

namespace NeuralNetwork
{
	class NeuralNetwork
	{
	public:
		NeuralNetwork(NeuronSettings *settings);
		~NeuralNetwork();

		inline double sigmoid(double value) { return 1.0f / (1.0f + exp(-value)); }
		inline double sigmoidPrime(double value) { return value * (1.0f - value); }
		inline double align(double value) { return sigmoid(value); }
		inline double alignPrime(double value) { return sigmoidPrime(value); }

		double* execute(double* input);

		void train(TrainSet* trainSet);

		void print();

	private:
		void _trainByLayer(int layer);

	private:
		NeuronLayer** _layers;
		int _layerCount;

		static int INPUT_LAYER;
		int OUTPUT_LAYER;

		double _alpha;

		//Chached data
		double** _deltas;
		double** _layersOut;
	}; // class NeuralNetwork
} //namespace NeuralNetwork

#endif // NEURAL_NETWORK_H
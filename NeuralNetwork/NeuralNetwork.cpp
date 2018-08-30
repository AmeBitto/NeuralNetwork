#include "NeuralNetwork.h"

#include <iostream>
#include <chrono>

namespace NeuralNetwork
{
	int NeuralNetwork::INPUT_LAYER = 0;

	NeuralNetwork::NeuralNetwork(NeuronSettings *settings)
	{
		_alpha = settings->getLearningRate();

		_layerCount = settings->getLayerCount();
		OUTPUT_LAYER = _layerCount;
		// Input layer without delta
		_deltas = new double*[_layerCount];
		_layerCount++; // plus input layer
		_layers = new NeuronLayer*[_layerCount];
		_layersOut = new double*[_layerCount];

		// Setup input layer
		_layers[INPUT_LAYER] = new NeuronLayer(settings->getInputCount(), settings->getInputCount(), settings->isInputBiasUsed());
		NeuronLayer *inputLayerPtr = _layers[INPUT_LAYER];
		int inputNeuronCount = settings->getInputCount();
		if (settings->isInputBiasUsed()) inputNeuronCount++;
		_layersOut[INPUT_LAYER] = new double[inputLayerPtr->getNeuronCount()];
		for (int n = 0; n < inputNeuronCount; n++)
		{
			Neuron* inputNeuronPtr = inputLayerPtr->getNeurons()[n];
			if (inputNeuronPtr->isBias()) continue;
			for (int w = 0; w < inputNeuronPtr->getWeightCount(); w++)
			{
				inputNeuronPtr->setWeight(w, (n == w) ? 1.0f : 0.0f);
			}
		}

		// Setup hidden layers
		for (int l = 1; l < _layerCount; l++)
		{
			NeuronLayerSettings* layer = settings->getLayers()[l - 1];
			_layers[l] = new NeuronLayer(_layers[l - 1]->getNeuronCount(), layer->getNeuronCount(), layer->isBiasUsed());
			NeuronLayer *layerPtr = _layers[l];
			int neuronsCount = layer->getNeuronCount();
			if (layer->isBiasUsed()) neuronsCount++;
			_layersOut[l] = new double[layerPtr->getNeuronCount()];
			_deltas[l - 1] = new double[layer->getNeuronCount()];
			if (layer->getWeights())
			{
				layerPtr->setupWeights(layer->getWeights());
			}
			else
			{
				layerPtr->randomWeights();
			}
		}
	}

	NeuralNetwork::~NeuralNetwork()
	{
		for (int l = 0; l < _layerCount; l++)
		{
			delete _layers[l];
			delete _layersOut[l];
			if (l < _layerCount - 1) delete[] _deltas[l];
		}

		delete[] _deltas;
		delete[] _layers;
		delete[] _layersOut;
	}

	double* NeuralNetwork::execute(double* input)
	{
		// Input
		NeuronLayer* inputLayerPtr = _layers[INPUT_LAYER];
		double* inputLayerOutPtr = _layersOut[INPUT_LAYER];
		int inputNeuronCount = inputLayerPtr->getNeuronCount();
		
		for (int n = 0; n < inputNeuronCount; n++)
		{
			inputLayerOutPtr[n] = inputLayerPtr->getNeurons()[n]->getOutput(input);
		}
		
		// Hidden
		for (int l = 1; l < _layerCount; l++)
		{
			NeuronLayer* layerPtr = _layers[l];
			double* layerOutPtr = _layersOut[l];
			int neuronCount = layerPtr->getNeuronCount();
			for (int n = 0; n < neuronCount; n++)
			{
				Neuron* neuronPtr = layerPtr->getNeurons()[n];
				double net = neuronPtr->getOutput(_layersOut[l - 1]);
				layerOutPtr[n] = (neuronPtr->isBias()) ? net : align(net);
				neuronPtr->setMemory(layerOutPtr[n]);
			}
		}

		return _layersOut[OUTPUT_LAYER];
	}

	void NeuralNetwork::train(TrainSet* trainSet)
	{
		//std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		double* answer = execute(trainSet->getInput());
		//std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		//auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
		//test += duration;

		NeuronLayer* outputLayerPtr = _layers[OUTPUT_LAYER];
		double* outputDeltaPtr = _deltas[OUTPUT_LAYER - 1];
		int neuronCount = outputLayerPtr->getNeuronCount();
		if (outputLayerPtr->isBiasUsed()) neuronCount--;

		for (int n = 0; n < neuronCount; n++)
		{
			outputDeltaPtr[n] = -(trainSet->getOutput()[n] - answer[n]) * alignPrime(answer[n]);
		}
		
		_trainByLayer(OUTPUT_LAYER);

	}

	void NeuralNetwork::_trainByLayer(int layer)
	{
		NeuronLayer* layerPtr = _layers[layer];
		double* deltaPtr = _deltas[layer - 1];
		NeuronLayer* previousLayerPtr = _layers[layer - 1];
		
		int neuronCount = layerPtr->getNeuronCount();
		if (layerPtr->isBiasUsed()) neuronCount--;

		int previousNeuronCount = previousLayerPtr->getNeuronCount();
		if (previousLayerPtr->isBiasUsed()) previousNeuronCount--;
		
		for (int n = 0; n < neuronCount; n++)
		{
			double* oldNeuronWeights = layerPtr->getNeurons()[n]->getWeights();
			double* newNeuronWeights = layerPtr->getNeurons()[n]->getChachedWeights();
			
			for (int w = 0; w < previousNeuronCount; w++)
			{
				newNeuronWeights[w] = oldNeuronWeights[w] - _alpha * deltaPtr[n] * previousLayerPtr->getNeurons()[w]->getMemory();
			}
		}


		// Next layer
		if (layer > 1)
		{	
			double* previousDeltaPtr = _deltas[layer - 2];

			for (int n = 0; n < previousNeuronCount; n++)
			{
				double deltaH = 0.0f;
				for (int d = 0; d < neuronCount; d++)
				{
					deltaH += deltaPtr[d] * layerPtr->getNeurons()[d]->getWeights()[n];
				}
				previousDeltaPtr[n] = deltaH * alignPrime(previousLayerPtr->getNeurons()[n]->getMemory());
			}

			_trainByLayer(layer - 1);
		}

		_layers[layer]->loadWeightsFromChache();
	}

	void NeuralNetwork::print()
	{
		printf("\nNEURAL NETWORK:\n");
		for (int l = 0; l < _layerCount; l++)
		{
			int neuronCount = _layers[l]->getNeuronCount();
			printf("\tLAYER %d: %d neurons\n", l, neuronCount);
			for (int n = 0; n < neuronCount; n++)
			{
				int weightCount = _layers[l]->getNeurons()[n]->getWeightCount();
				printf("\t\tNEURON %d: %d weights\n", n, weightCount);
				for (int w = 0; w < weightCount; w++)
				{
					printf("\t\t\tWEIGHT %d: %f\n", w, _layers[l]->getNeurons()[n]->getWeights()[w]);
				}
			}
		}
		printf("\nNEURAL NETWORK END\n");
	}

} // namespace NeuralNetwork
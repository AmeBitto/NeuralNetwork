#include <iostream>
#include <ctime>

#include "NeuralNetwork/NeuralNetwork.h"

void study(NeuralNetwork::NeuralNetwork* brain, int cicles)
{
	clock_t begin = clock();

	// TrainSet 1
	double* input1 = new double[2]; input1[0] = 1.0f; input1[1] = 0.0f;
	double* output1 = new double[1]; output1[0] = 1.0f;
	NeuralNetwork::TrainSet* trainSet1 = new NeuralNetwork::TrainSet(input1, output1);

	// TrainSet 2
	double* input2 = new double[2]; input2[0] = 0.0f; input2[1] = 1.0f;
	double* output2 = new double[1]; output2[0] = 1.0f;
	NeuralNetwork::TrainSet* trainSet2 = new NeuralNetwork::TrainSet(input2, output2);

	// TrainSet 3
	double* input3 = new double[2]; input3[0] = 0.0f; input3[1] = 0.0f;
	double* output3 = new double[1]; output3[0] = 0.0f;
	NeuralNetwork::TrainSet* trainSet3 = new NeuralNetwork::TrainSet(input3, output3);

	// TrainSet 4
	double* input4 = new double[2]; input4[0] = 1.0f; input4[1] = 1.0f;
	double* output4 = new double[1]; output4[0] = 0.0f;
	NeuralNetwork::TrainSet* trainSet4 = new NeuralNetwork::TrainSet(input4, output4);

	// Train
	for (int c = 0; c < cicles; c++)
	{
		brain->train(trainSet1);
		brain->train(trainSet2);
		brain->train(trainSet3);
		brain->train(trainSet4);
	}

	// Free memory
	delete[] input1;
	delete[] output1;
	delete trainSet1;

	delete[] input2;
	delete[] output2;
	delete trainSet2;

	delete[] input3;
	delete[] output3;
	delete trainSet3;

	delete[] input4;
	delete[] output4;
	delete trainSet4;

	clock_t end = clock();

	printf("Study took: %d msec\n", end - begin);
}

int main(int argc, char* argv[])
{
	NeuralNetwork::NeuronSettings* settings = 
		new NeuralNetwork::NeuronSettings(0.5f, 4, 2);

	// Hidden layer
	NeuralNetwork::NeuronLayerSettings* hiddenLayer1 =
		new NeuralNetwork::NeuronLayerSettings(4);
	settings->addLayer(hiddenLayer1);
	NeuralNetwork::NeuronLayerSettings* hiddenLayer2 =
		new NeuralNetwork::NeuronLayerSettings(10);
	settings->addLayer(hiddenLayer2);
	NeuralNetwork::NeuronLayerSettings* hiddenLayer3 =
		new NeuralNetwork::NeuronLayerSettings(8);
	settings->addLayer(hiddenLayer3);

	// Output layer
	NeuralNetwork::NeuronLayerSettings* outputLayer =
		new NeuralNetwork::NeuronLayerSettings(1);
	settings->addLayer(outputLayer);

	// Create brain
	NeuralNetwork::NeuralNetwork* brain = 
		new NeuralNetwork::NeuralNetwork(settings);

	// Free settings
	delete settings;
	delete hiddenLayer1;
	delete hiddenLayer2;
	delete hiddenLayer3;
	delete outputLayer;

	brain->print();

	// Study
	study(brain, 1000000);

	brain->print();

	// Test 1
	double* test1 = new double[2]; test1[0] = 1.0f; test1[1] = 0.0f;
	double* answer = brain->execute(test1);
	printf("test[1, 0] - waiting: 1; answer: %f\n", answer[0]);
	delete[] test1;

	// Test 2
	double* test2 = new double[2]; test2[0] = 0.0f; test2[1] = 1.0f;
	answer = brain->execute(test2);
	printf("test[0, 1] - waiting: 1; answer: %f\n", answer[0]);
	delete[] test2;


	// Test 3
	double* test3 = new double[2]; test3[0] = 0.0f; test3[1] = 0.0f;
	answer = brain->execute(test3);
	printf("test[0, 0] - waiting: 0; answer: %f\n", answer[0]);
	delete[] test3;

	// Test 4
	double* test4 = new double[2]; test4[0] = 1.0f; test4[1] = 1.0f;
	answer = brain->execute(test4);
	printf("test[1, 1] - waiting: 0; answer: %f\n", answer[0]);
	delete[] test4;

	system("pause");
	return 0;
}
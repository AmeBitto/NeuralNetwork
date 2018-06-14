#ifndef TRAIN_SET_H
#define TRAIN_SET_H

namespace NeuralNetwork
{
	class TrainSet
	{
	public:
		TrainSet(double* input, double* output);
		~TrainSet();

		inline double* getInput() const { return _input; }
		inline double* getOutput() const { return _output; }

	private:
		double* _input;
		double* _output;
	}; // class TrainSet
} // namespace NeuralNetwork

#endif // TRAIN_SET_H
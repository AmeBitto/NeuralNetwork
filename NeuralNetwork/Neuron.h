#ifndef NEURON_H
#define NEURON_H

namespace NeuralNetwork
{
	class Neuron
	{
	public:
		Neuron(int inputCount, bool isBias = false);
		~Neuron();

		// Bias
		inline bool isBias() const { return _isBias; }

		// Weights
		inline double* getWeights() const { return _weights; }
		void setWeights(double* weights);
		void setWeight(int index, double weight);

		inline double* getChachedWeights() const { return _chachedWeights; }
		void setChachedWeights(double* weights);
		void setChachedWeight(int index, double weight);

		inline int getWeightCount() const { return _weightCount; }	

		// Speed up
		void loadWeightsFromChache();

		// Result
		double getOutput(double* input);
		
		// Memory
		inline void setMemory(double value) { _memory = value; }
		inline double getMemory() const { return _memory; }

	private:
		bool _isBias;
		double* _weights;
		double* _chachedWeights;
		int _weightCount;
		
		double _memory;
	}; // class Neuron
} // namespace NeuralNetwork

#endif // NEURON_H
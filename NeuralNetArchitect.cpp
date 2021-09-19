#include <cstdlib>
#include <random>
#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <vector>
#include <assert.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

/**********************************************************************************************************************************************
 Neuron's activation is the sumOfproducts(weights, inputActivations) + bias, or the given input if it is in the input layer

  neuronInputListCount; number of neurons with activation that feeds into this neuron's activation function
  inputNeurons; array of addresses of input neurons with an activation that forms part of this neuron's activation
  activation; the evaluation of this neuron's activation function: sumOfproducts(weights, inputActivations) + bias
  activationNudgeSum; measurement of how this activation affects cost function, found by sum (dC/da)*(da/da_this) from proceeding neurons
  weights; array of learned weights that are used to modify impact of input neuron activations on this neuron's activation
  weightsMomentum; The momentum of weights being updated by the previous nudge, which will have an effect on subsequent nudges
  bias; the learned negative of the activation threshold that the sumOfProducts needs to surpass to have a positive activation
  biasMomentum; The momentum of the bias being updated by the previous nudge, which will have an effect on subsequent nudges
  momentumRetention; The inverse rate of decay of a parameter's momentum having an effect in next nudge. if 0, no impact.
 **********************************************************************************************************************************************/
class Neuron
{

private:
	int neuronInputListCount;
	Neuron* inputNeurons;
	double activation, activationNudgeSum;
	double* weights, * weightsMomentum;
	double bias, biasMomentum;
	double momentumRetention;

protected:
	//Computes neuron's internal sumproduct, weights*input activations and bias
	double getActivationFunctionInput() const
	{
		double sumOfProduct = 0;
		for (auto i = 0; i < neuronInputListCount; i++)
		{
			sumOfProduct += weights[i] * inputNeurons[i].getActivation();
		}

		return sumOfProduct + bias;
	}

	//returns the current calculation for derivative of cost function in respect to this neuron's activation
	double getActivationNudgeSum() const
	{
		return activationNudgeSum;
	}

	//Calculates partial derivative of cost function in respect to indexed input neuron activation: dC/da * da/di = dC/di
	virtual double getActivationRespectiveDerivation(const int inputNeuronIndex) const
	{
		assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

		return activationNudgeSum * weights[inputNeuronIndex];
	}

	//Calculates partial derivative of cost function in respect to indexed weight: dC/da * da/dw = dC/dw
	virtual double getWeightRespectiveDerivation(const int inputNeuronIndex) const
	{
		assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

		return activationNudgeSum * inputNeurons[inputNeuronIndex].getActivation();
	}

	//Calculates partial derivative of cost function in respect to indexed input neuron activation: dC/da * da/db = dC/db
	virtual double getBiasRespectiveDerivation() const
	{
		assert(neuronInputListCount >= 0);

		return activationNudgeSum * 1.0;
	}

	//Adds desired change in activation value that would've reduced minibatch training error, dC/da = completeSum(dC/do * do/da)
	void nudgeActivation(double nudge)
	{
		activationNudgeSum += nudge;
	}

public:
	//constructor called for input neurons of activation determined by input
	Neuron() : weights(nullptr), weightsMomentum(nullptr), inputNeurons(nullptr)
	{
		this->neuronInputListCount = 0;
		this->momentumRetention = 0;

		bias = biasMomentum = 0.0;

		activation = activationNudgeSum = 0.0;
	}

	//constructor called for hidden neurons during network creation, with optional learning momentum parameter
	Neuron(int neuronInputListCount, Neuron* inputNeurons, double momentumRetention = 0.0)
	{
		this->neuronInputListCount = neuronInputListCount;
		this->inputNeurons = inputNeurons;
		this->momentumRetention = momentumRetention;

		//Initialize tools for randomly generating numbers that follow a gaussian distribution
		std::random_device randomDevice{};
		std::mt19937 generator{ randomDevice() };
		std::normal_distribution<double> randomGaussianDistributor{ 0.0, std::sqrt(2 / (double)neuronInputListCount) };

		//Initializes weights using He-et-al method
		weights = new double[neuronInputListCount];
		if (weights == nullptr) throw std::bad_alloc();
		for (auto i = 0; i < neuronInputListCount; i++)
		{
			weights[i] = randomGaussianDistributor(generator);
		}

		weightsMomentum = new double[neuronInputListCount]();
		if (weightsMomentum == nullptr) throw std::bad_alloc();

		bias = biasMomentum = 0.0;

		activation = activationNudgeSum = 0.0;
	}

	//constructor called for hidden neurons during network loading, with stored weights and bias values passed in
	Neuron(int neuronInputListCount, Neuron* inputNeurons, std::vector<double> weightValues, double biasValue, double momentumRetention = 0.0)
	{
		this->neuronInputListCount = neuronInputListCount;
		this->inputNeurons = inputNeurons;
		this->momentumRetention = momentumRetention;

		//Initializes weights using He-et-al method
		weights = new double[neuronInputListCount];
		if (weights == nullptr) throw std::bad_alloc();
		for (auto i = 0; i < neuronInputListCount; i++)
			weights[i] = weightValues[i];

		weightsMomentum = new double[neuronInputListCount]();
		if (weightsMomentum == nullptr) throw std::bad_alloc();

		bias = biasValue;
		biasMomentum = 0.0;

		activation = activationNudgeSum = 0.0;
	}

	//copy constructor for neurons
	Neuron(const Neuron& original)
	{
		neuronInputListCount = original.neuronInputListCount;
		inputNeurons = original.inputNeurons;
		activation = original.activation;
		activationNudgeSum = original.activationNudgeSum;
		bias = original.bias;
		biasMomentum = original.biasMomentum;
		momentumRetention = original.momentumRetention;

		weights = new double[neuronInputListCount];
		if (weights == nullptr) throw std::bad_alloc();
		for (auto i = 0; i < neuronInputListCount; i++)
			weights[i] = original.weights[i];

		weightsMomentum = new double[neuronInputListCount];
		if (weightsMomentum == nullptr) throw std::bad_alloc();
		for (auto i = 0; i < neuronInputListCount; i++)
			weightsMomentum[i] = original.weightsMomentum[i];
	}

	//operator = overloading for readable assignments resulting in deep copies
	Neuron& operator=(const Neuron& original)
	{
		neuronInputListCount = original.neuronInputListCount;
		inputNeurons = original.inputNeurons;
		activation = original.activation;
		activationNudgeSum = original.activationNudgeSum;
		bias = original.bias;
		biasMomentum = original.biasMomentum;
		momentumRetention = original.momentumRetention;

		weights = new double[neuronInputListCount];
		if (weights == nullptr) throw std::bad_alloc();
		for (auto i = 0; i < neuronInputListCount; i++)
			weights[i] = original.weights[i];

		weightsMomentum = new double[neuronInputListCount];
		if (weightsMomentum == nullptr) throw std::bad_alloc();
		for (auto i = 0; i < neuronInputListCount; i++)
			weightsMomentum[i] = original.weightsMomentum[i];

		return *this;
	}

	//custom destructor for neurons
	~Neuron()
	{
		inputNeurons = nullptr;

		delete[] weights;
		delete[] weightsMomentum;
	}

	//Defines empty exterior activation function of neuron, a linear sumOfProducts(weights,inputActivations) + bias
	virtual void activate(const double input = 0.0)
	{
		if (neuronInputListCount > 0)
		{
			activation = getActivationFunctionInput();
		}
		else
		{
			activation = input;
		}

	}

	//Injects error dC/da into neuron
	void setError(double cost)
	{
		activationNudgeSum = cost;
	}

	//Injects corresponding error into input neurons due to activation, dC/di = sum(all(dC/dh * dh/di)) 
	void injectInputRespectiveCostDerivation() const
	{
		for (auto i = 0; i < neuronInputListCount; i++)
		{
			inputNeurons[i].nudgeActivation(getActivationRespectiveDerivation(i));
		}
	}

	//Applies change to weights that would reduce cost for past batch - uses reserved activationNudges to scale change proportionally
	void updateWeights(int batchSize, double learningRate)
	{
		for (auto i = 0; i < neuronInputListCount; i++)
		{
			weightsMomentum[i] = momentumRetention * weightsMomentum[i] - (getWeightRespectiveDerivation(i) / batchSize) * learningRate;
			weights[i] += weightsMomentum[i];
		}
	}

	//Applies change to bias that would reduce cost function for past batch - uses reserved activationNudges to scale change proportionally
	void updateBias(int batchSize, double learningRate)
	{
		biasMomentum = momentumRetention * biasMomentum - (getBiasRespectiveDerivation() / batchSize) * learningRate;
		bias += biasMomentum;
	}

	//Resets partial derivative of cost in respect to this neuron's activation from past batch
	void resetNudges()
	{
		activationNudgeSum = 0.0;
	}

	//returns number of input neurons
	int getInputCount() const
	{
		return neuronInputListCount;
	}

	//returns activation value of neuron
	double getActivation() const
	{
		return activation;
	}

	//returns weight from this neuron towards a specified input neuron
	double getWeight(int inputNeuronIndex) const
	{
		assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

		return weights[inputNeuronIndex];
	}

	std::vector<double> getWeights() const
	{
		std::vector<double> weights;

		for (auto i = 0; i < neuronInputListCount; i++)
		{
			weights.push_back(getWeight(i));
		}

		return weights;
	}

	//returns bias of this neuron
	double getBias() const
	{
		return bias;
	}

	//returns the activation type of the neuron
	virtual std::string getNeuronType()
	{
		return getInputCount() == 0 ? "Input" : "Linear";
	}

};

/**********************************************************************************************************************************************
 NeuralLayer's activation is strictly function(sumOfproducts(weights, inputActivations) + biases) or input array, with no additional variables

  neuronArrayLength; number of neurons contained within each column of a layer
  neuronArrayWidth; number of neurons contained within each row of a layer
  neurons; array of neurons contained within layer
  previousLayer; a pointer to the NeuralLayer that is to feed into this layer - nullptr if this is first layer
 **********************************************************************************************************************************************/
class NeuralLayer
{

protected:
	int neuronArrayLength, neuronArrayWidth;
	Neuron* neurons;
	NeuralLayer* previousLayer;

	//Set error of neurons with activations directly used to calculate cost dC/da
	void setError(double costArray[])
	{
		if (costArray != nullptr)
		{
			for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
				neurons[i].setError(costArray[i]);
		}
	}

	//nudge input layer activations with appropriate derivatives of cost function dC/da * da/di
	void injectErrorBackwards()
	{
		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
			neurons[i].injectInputRespectiveCostDerivation();
	}

	//apply learned weights and bias updates
	void updateParameters(int batchSize, double learningRate)
	{
		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons[i].updateWeights(batchSize, learningRate);

			neurons[i].updateBias(batchSize, learningRate);
		}
	}

	//clears all stored nudges to neuron parameters
	void clearNudges()
	{
		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
			neurons[i].resetNudges();
	}

public:
	//default constructor for layer class
	NeuralLayer()
	{
		neurons = nullptr;
		neuronArrayLength = 0;
		neuronArrayWidth = 0;
		previousLayer = nullptr;
	}

	//constructor called for input layers
	NeuralLayer(int inputLength, int inputWidth) : neuronArrayLength(inputLength), neuronArrayWidth(inputWidth), previousLayer(nullptr)
	{
		neurons = new Neuron[inputLength * inputWidth];
		if (neurons == nullptr) throw std::bad_alloc();

		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons[i] = Neuron();
		}
	}

	//constructor called for hidden layers during network creation, with optional momentum parameter
	NeuralLayer(int neuronCount, NeuralLayer* inputLayer, double momentumRetention = 0.0)
	{
		neuronArrayLength = neuronCount;
		neuronArrayWidth = 1;
		previousLayer = inputLayer;

		int inputNeuronCount = previousLayer->getNeuronArrayCount();
		Neuron* inputNeurons = previousLayer->getNeurons();
		neurons = new Neuron[neuronCount];
		if (neurons == nullptr) throw std::bad_alloc();

		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons[i] = Neuron(inputNeuronCount, inputNeurons, momentumRetention);
		}

	}

	//constructor called for hidden layers during network loading, with stored weights and bias values passed in
	NeuralLayer(int neuronCount, NeuralLayer* inputLayer, double momentumRetention, std::vector<std::vector<double>> weightValues, std::vector<double> biasValues)
	{
		neuronArrayLength = neuronCount;
		neuronArrayWidth = 1;
		previousLayer = inputLayer;

		int inputNeuronCount = previousLayer->getNeuronArrayCount();
		Neuron* inputNeurons = previousLayer->getNeurons();
		neurons = new Neuron[neuronCount];
		if (neurons == nullptr) throw std::bad_alloc();

		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons[i] = Neuron(inputNeuronCount, inputNeurons, weightValues[i], biasValues[i], momentumRetention);
		}

	}

	//copy constructor for layers
	NeuralLayer(const NeuralLayer& original)
	{
		neuronArrayLength = original.neuronArrayLength;
		neuronArrayWidth = original.neuronArrayWidth;
		previousLayer = original.previousLayer;

		neurons = new Neuron[neuronArrayLength * neuronArrayWidth];
		if (neurons == nullptr) throw std::bad_alloc();

		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons[i] = Neuron(original.neurons[i]);
		}
	}

	//operator = overloading for readable assignments resulting in deep copies
	NeuralLayer& operator=(const NeuralLayer& original)
	{
		neuronArrayLength = original.neuronArrayLength;
		neuronArrayWidth = original.neuronArrayWidth;
		previousLayer = original.previousLayer;

		neurons = new Neuron[neuronArrayLength * neuronArrayWidth];
		if (neurons == nullptr) throw std::bad_alloc();

		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons[i] = Neuron(original.neurons[i]);
		}

		return (*this);
	}

	//custom destructor for NeuralLayer objects
	~NeuralLayer()
	{
		delete[] neurons;

		previousLayer = nullptr;
	}

	//activate all neurons in layer and resets nudges from past learning iteration
	void propagateForward(double inputValues[] = nullptr)
	{
		if (previousLayer == nullptr)
		{
			for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
			{
				neurons[i].activate(inputValues[i]);
			}
		}

		else
		{
			for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
			{
				neurons[i].activate();
			}
		}

		clearNudges();
	}

	//transmit error to input neurons and apply learned parameter updates
	void propagateBackward(int batchSize, double learningRate, double* costArray = nullptr)
	{
		setError(costArray);

		injectErrorBackwards();

		updateParameters(batchSize, learningRate);
	}

	//returns number of neurons contained within a column of the layer
	int getNeuronArrayLength() const
	{
		return neuronArrayLength;
	}

	//returns number of neurons contained within a row of the layer
	int getNeuronArrayWidth() const
	{
		return neuronArrayWidth;
	}

	//returns number of neurons contained within layer
	int getNeuronArrayCount() const
	{
		return getNeuronArrayLength() * getNeuronArrayWidth();
	}

	//returns array of pointers to neurons contained within layer
	Neuron* getNeurons() const
	{
		return neurons;
	}

	//returns pointer to layer that is feeding into this layer
	NeuralLayer* getPreviousLayer() const
	{
		return previousLayer;
	}

	std::vector<double> getNeuronActivations() const
	{
		std::vector<double> neuronActivations;

		for (auto i = 0; i < getNeuronArrayCount(); i++)
		{
			neuronActivations.push_back(getNeurons()[i].getActivation());
		}

		return neuronActivations;
	}

	std::vector<std::vector<double>> getNeuronWeights() const
	{
		std::vector<std::vector<double>> neuronWeights;

		for (auto i = 0; i < getNeuronArrayCount(); i++)
		{
			neuronWeights.push_back(getNeurons()[i].getWeights());
		}

		return neuronWeights;
	}

	std::vector<double> getNeuronBiases() const
	{
		std::vector<double> neuronBiases;

		for (auto i = 0; i < getNeuronArrayCount(); i++)
		{
			neuronBiases.push_back(getNeurons()[i].getBias());
		}

		return neuronBiases;
	}

	//returns the activation type of the neurons contained within layer
	virtual int getNeuralLayerType() const
	{
		return 1;
	}
};

//the derivation of the mean-squared-error function in respect to the activation of an output neuron
double derivedMSECost(double targetValue, double estimatedValue, int outputCount)
{
	return (-2 / outputCount) * (targetValue - estimatedValue);
}

struct layerCreationInfo
{
	int type;
	int neuronCount;
	double momentumRetention;
};

struct layerLoadingInfo
{
	int type;
	int neuronCount;
	double momentumRetention;
	std::vector<std::vector<double>> weightsOfNeurons;
	std::vector<double> biasOfNeurons;
};

/**********************************************************************************************************************************************
 NeuralNetworks's activation is a function of all weights and bias parameters held within the neurons of each layer

  layerCount; number of neural layers held within neural network (also defines the depth of the network)
  inputLength; the first dimension defining the size of the input array, currently assuming a 2D input grid
  inputWidth; the first dimension defining the size of the input array, currently assuming a 2D input grid
  outputCount; the number of outputs the neural network is expected to produce, currently assuming a vector output
  neuralLayers; an array containing all neural layers that make up the network
  learningRate; coefficient describing the magnitude of the adjustments to weight and bias parameters following a training iteration
  batchSize; number of training samples from a dataset that will be fed-forward through the network before learning takes place
 **********************************************************************************************************************************************/
class NeuralNetwork
{

private:
	int layerCount;
	int inputLength, inputWidth;
	int outputCount;
	NeuralLayer* neuralLayers;
	double learningRate;
	int batchSize;
	double (*derivedCostFunction)(double, double, int);
	layerLoadingInfo* layerStates;

public:
	//constructor for creating NeuralNetworks
	NeuralNetwork(int layerCount, int inputLength, int inputWidth, int outputCount, double learningRate, int batchSize, int costSelection, layerCreationInfo* layerDetails)
	{
		this->layerCount = layerCount;
		this->inputLength = inputLength;
		this->inputWidth = inputWidth;
		this->outputCount = outputCount;
		this->learningRate = learningRate;
		this->batchSize = batchSize;
		
		switch (costSelection)
		{
		case 1:
			this->derivedCostFunction = derivedMSECost;
			break;
		default:
			this->derivedCostFunction = derivedMSECost;
			break;
		}

		neuralLayers = new NeuralLayer[layerCount];
		if (neuralLayers == nullptr) throw std::bad_alloc();
		neuralLayers[0] = NeuralLayer(inputLength, inputWidth);

		for (auto i = 1; i < layerCount; i++)
		{
			switch (layerDetails[i].type)
			{
			case 1:
				this->neuralLayers[i] = NeuralLayer(layerDetails[i].neuronCount, &neuralLayers[i - 1], layerDetails[i].momentumRetention);
				break;
			default:
				this->neuralLayers[i] = NeuralLayer(layerDetails[i].neuronCount, &neuralLayers[i - 1], layerDetails[i].momentumRetention);
				break;
			}
		}

		layerStates = new layerLoadingInfo[layerCount];
	}

	//todo: create load constructor

	//returns a vector of the activation values of the final layer of the network
	std::vector<double> getOutputs()
	{
		return neuralLayers[layerCount - 1].getNeuronActivations();
	}

	//activates all layers in order from input to output layers
	void propagateForwards(double* inputMatrix)
	{
		neuralLayers[0].propagateForward(inputMatrix);

		for (auto i = 1; i < layerCount; i++)
		{
			neuralLayers[i].propagateForward();
		}
	}

	//updates parameters in all layers in order from output to input layers
	void propagateBackwards(double* costArray)
	{
		neuralLayers[layerCount - 1].propagateBackward(batchSize, learningRate, costArray);

		for (auto i = layerCount - 2; i > 0; i--)
		{
			neuralLayers[i].propagateBackward(batchSize, learningRate);
		}
	}

	//changes number of samples network expects to process before being told to learn
	void updateBatchSize(int newBatchSize)
	{
		batchSize = newBatchSize;
	}

	//updates magnitude of parameter changes during learning
	void updateLearningRate(int newLearningRate)
	{
		learningRate = newLearningRate;
	}

	//gives the partial derivative value of the cost function in respect to an output activation
	double getOutputRespectiveCost(double targetValue, int outputIndex)
	{
		return derivedCostFunction(targetValue, getOutputs()[outputIndex], outputCount);
	}

	//gives the number of inputs that network accepts
	int getInputCount()
	{
		return inputLength * inputWidth;
	}

	//gives the number of outputs that the network produces
	int getOutputCount()
	{
		return outputCount;
	}

	//gives the depth of the network
	int getLayerCount()
	{
		return layerCount;
	}

	//saves all necessary layer data necessary to recreate the network layers exactly upon loading
	void saveLayerStates()
	{
		for (auto i = 0; i < getLayerCount(); i++)
		{
			layerStates[i].type = neuralLayers[i].getNeuralLayerType();
			layerStates[i].neuronCount = neuralLayers[i].getNeuronArrayCount();
			layerStates[i].momentumRetention = 0;
			layerStates[i].weightsOfNeurons = neuralLayers[i].getNeuronWeights();
			layerStates[i].biasOfNeurons = neuralLayers[i].getNeuronBiases();
		}
	}

	//gives array of layer state details that may be used to recreate layers
	layerLoadingInfo* getLayerStates()
	{
		return layerStates;
	}
};

//saves the entire neural network to an xml, such that all data necessary to rebuild the exact network is stored
void storeNetwork(NeuralNetwork network, std::string &fileName)
{
	int inputLength, outputLength, networkDepth, optimizationAlgorithm, errorFunction;

	//saves network details that are not specific to the layers
	inputLength = network.getInputCount();
	outputLength = network.getOutputCount();
	networkDepth = network.getLayerCount();
	optimizationAlgorithm = 0;
	errorFunction = 0;

	//initializes array of fully defined layer states
	network.saveLayerStates();
	layerLoadingInfo* layerStates = network.getLayerStates();

	//defines property trees and subtrees
	boost::property_tree::ptree networkPropertyTree, layerPropertySubTree, neuronPropertySubTree;

	//adds non-layer network details as children to property tree root
	networkPropertyTree.put("network.inputLength", inputLength);
	networkPropertyTree.put("network.outputLength", outputLength);
	networkPropertyTree.put("network.networkDepth", networkDepth);
	networkPropertyTree.put("network.optimizationAlgorithm", optimizationAlgorithm);
	networkPropertyTree.put("network.errorFunction", errorFunction);

	//defines and inserts layer detail subtrees as children to network ptree's 'layers' member
	for (auto i = 0; i < networkDepth; i++)
	{
		//adds non-neuron layer details as chidlren to property subtree root
		layerPropertySubTree.put("activationType", layerStates[i].type);
		layerPropertySubTree.put("neuronCount", layerStates[i].neuronCount);
		layerPropertySubTree.put("momentumRetention", layerStates[i].momentumRetention);

		//defines and inserts neuron detail subtrees as children to layer ptree's 'neurons' member
		for (auto j = 0; j < layerStates[i].neuronCount; j++)
		{
			//adds neuron's current bias parameter value as a child to its property subtree root
			neuronPropertySubTree.put("bias", layerStates[i].biasOfNeurons[j]);

			//adds neuron's current weight parameter values as children to its property subtree root
			for (std::vector<double>::iterator it = layerStates[i].weightsOfNeurons[j].begin(); it < layerStates[i].weightsOfNeurons[j].end(); it++)
			{
				neuronPropertySubTree.add("weights.weight", (*it));
			}
			
			//inserts fully-defined neuron subtree as child to layer subtree's 'neurons' member and clears neuron subtree
			layerPropertySubTree.add_child("neurons.neuron", neuronPropertySubTree);
			neuronPropertySubTree.clear();
		}

		//inserts fully-defined layer subtree as child to network tree's 'layers' member and clears layer subtree
		networkPropertyTree.add_child("network.layers.layer", layerPropertySubTree);
		layerPropertySubTree.clear();
	}

	//creates xml file of fully-defined neural network structure and parameters
	boost::property_tree::write_xml(fileName, networkPropertyTree);
}

int main()
{

	int numberOfLayers, inputLength, inputWidth, outputCount, batchSize, costSelection;

	std::cout << "What is the length of inputs that this neural network will accept? ";
	std::cin >> inputLength;
	std::cout << std::endl;

	//std::cout << "What is the width of inputs that this neural network will accept? ";
	//std::cin >> inputWidth;
	inputWidth = 1;
	//std::cout << std::endl;

	std::cout << "What is the number of outputs that this neural network will produce? ";
	std::cin >> outputCount;
	std::cout << std::endl;

	std::cout << "How many layers will this neural network contain? ";
	std::cin >> numberOfLayers;
	layerCreationInfo* layerDetails = new layerCreationInfo[numberOfLayers];
	std::cout << std::endl;

	//std::cout << "What is the current batch size that this network will train on? ";
	//std::cin >> batchSize;
	batchSize = 1;
	//std::cout << std::endl;

	//std::cout << "Which cost function should be used to calculate error? ;
	//std::cin >> costSelection;
	costSelection = 1;
	//std::cout << std::endl;

	layerDetails[0].type = 1;
	layerDetails[0].neuronCount = inputLength * inputWidth;
	layerDetails[0].momentumRetention = 0;

	for (int i = 1; i < numberOfLayers; i++)
	{
		std::cout << std::endl << "Define neural layer " << i + 1 << ":\n";

		std::cout << "\tActivation type: ";
		std::cin >> layerDetails[i].type;
		std::cout << std::endl;

		if (i + 1 < numberOfLayers)
		{
			std::cout << "\tNeuron count: ";
			std::cin >> layerDetails[i].neuronCount;
			std::cout << std::endl;
		}
		else
		{
			layerDetails[i].neuronCount = outputCount;
		}


		std::cout << "\tMomentum retention: ";
		std::cin >> layerDetails[i].momentumRetention;
		layerDetails[i].momentumRetention = 0;
		std::cout << std::endl;
	}

	//create network
	NeuralNetwork network = NeuralNetwork(numberOfLayers, inputLength, inputWidth, outputCount, 0.0001, batchSize, costSelection, layerDetails);
	//todo: learning rate heuristics?

	//load inputs with dummy data
	double* inputGrid = new double[inputLength * inputWidth];
	for (auto i = 0; i < inputLength * inputWidth; i++)
	{
		inputGrid[i] = 15;
	}

	//propagate forwards
	network.propagateForwards(inputGrid);

	//get outputs
	auto outputVector = network.getOutputs();
	for (std::vector<double>::iterator it = outputVector.begin(); it < outputVector.end(); it++)
	{
		std::cout << (*it) << " ";
	}

	//calculate error vector
	double* errorVector = new double[outputCount];
	for (auto i = 0; i < outputCount; i++)
	{//todo: Cost function would go here, default to partial dC/da of MSE Cost Function
		errorVector[i] = network.getOutputRespectiveCost(20,i);
	}

	network.propagateBackwards(errorVector);

	//propagate forwards
	network.propagateForwards(inputGrid);

	//get outputs
	outputVector = network.getOutputs();
	for (std::vector<double>::iterator it = outputVector.begin(); it < outputVector.end(); it++)
	{
		std::cout << (*it) << " ";
	}

	std::string xmlName = "test0.xml";
	storeNetwork(network,xmlName);

	return 0;
}
// 2 1 4 1 1 0 1 2 0 1 0
// 2 2 4 1 1 0 1 2 0 1 0
// 1 1 2 1 0 single non-input neuron
// 1 1 3 1 1 0 1 0 series
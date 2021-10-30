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
#include "NeuralNetArchitect.h"

//Computes neuron's internal sumproduct, weights*input activations and bias
double Neuron::getActivationFunctionInput() const
{
	double sumOfProduct = 0;
	for (auto i = 0; i < neuronInputListCount; i++)
	{
		sumOfProduct += weights[i] * inputNeurons[i].getActivation();
	}

	return sumOfProduct + bias;
}

//returns the current calculation for derivative of cost function in respect to this neuron's activation
double Neuron::getActivationNudgeSum() const
{
	return activationNudgeSum;
}

//Calculates partial derivative of cost function in respect to indexed input neuron activation: dC/da * da/di = dC/di
double Neuron::getActivationRespectiveDerivation(const int inputNeuronIndex) const
{
	assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

	return activationNudgeSum * weights[inputNeuronIndex];
}

//Calculates partial derivative of cost function in respect to indexed weight: dC/da * da/dw = dC/dw
double Neuron::getWeightRespectiveDerivation(const int inputNeuronIndex) const
{
	assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

	return activationNudgeSum * inputNeurons[inputNeuronIndex].getActivation();
}

//Calculates partial derivative of cost function in respect to indexed input neuron activation: dC/da * da/db = dC/db
double Neuron::getBiasRespectiveDerivation() const
{
	assert(neuronInputListCount >= 0);

	return activationNudgeSum * 1.0;
}

//Adds desired change in activation value that would've reduced minibatch training error, dC/da = completeSum(dC/do * do/da)
void Neuron::nudgeActivation(double nudge)
{
	activationNudgeSum += nudge;
}

//constructor called for input neurons of activation determined by input
Neuron::Neuron() : weights(nullptr), weightsMomentum(nullptr), inputNeurons(nullptr)
{
	this->neuronInputListCount = 0;

	bias = biasMomentum = 0.0;

	activation = activationNudgeSum = 0.0;
}

//constructor called for hidden neurons during network creation, with optional learning momentum parameter
Neuron::Neuron(int neuronInputListCount, Neuron* inputNeurons)
{
	this->neuronInputListCount = neuronInputListCount;
	this->inputNeurons = inputNeurons;

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
Neuron::Neuron(int neuronInputListCount, Neuron* inputNeurons, std::vector<double> weightValues, double biasValue)
{
	this->neuronInputListCount = neuronInputListCount;
	this->inputNeurons = inputNeurons;

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
Neuron::Neuron(const Neuron& original)
{
	neuronInputListCount = original.neuronInputListCount;
	inputNeurons = original.inputNeurons;
	activation = original.activation;
	activationNudgeSum = original.activationNudgeSum;
	bias = original.bias;
	biasMomentum = original.biasMomentum;

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
Neuron& Neuron::operator=(const Neuron& original)
{
	neuronInputListCount = original.neuronInputListCount;
	inputNeurons = original.inputNeurons;
	activation = original.activation;
	activationNudgeSum = original.activationNudgeSum;
	bias = original.bias;
	biasMomentum = original.biasMomentum;

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
Neuron::~Neuron()
{
	inputNeurons = nullptr;

	delete[] weights;
	delete[] weightsMomentum;
}

//Defines empty exterior activation function of neuron, a linear sumOfProducts(weights,inputActivations) + bias
void Neuron::activate(const double input)
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
void Neuron::setError(double cost)
{
	activationNudgeSum = cost;
}

//Injects corresponding error into input neurons due to activation, dC/di = sum(all(dC/dh * dh/di)) 
void Neuron::injectInputRespectiveCostDerivation() const
{
	for (auto i = 0; i < neuronInputListCount; i++)
	{
		inputNeurons[i].nudgeActivation(getActivationRespectiveDerivation(i));
	}
}

//Applies change to weights that would reduce cost for past batch - uses reserved activationNudges to scale change proportionally
void Neuron::updateWeights(int batchSize, double learningRate, double momentumRetention)
{
	for (auto i = 0; i < neuronInputListCount; i++)
	{
		weightsMomentum[i] = momentumRetention * weightsMomentum[i] - (getWeightRespectiveDerivation(i) / batchSize) * learningRate;
		weights[i] += weightsMomentum[i];
	}
}

//Applies change to bias that would reduce cost function for past batch - uses reserved activationNudges to scale change proportionally
void Neuron::updateBias(int batchSize, double learningRate, double momentumRetention)
{
	biasMomentum = momentumRetention * biasMomentum - (getBiasRespectiveDerivation() / batchSize) * learningRate;
	bias += biasMomentum;
}

//Resets partial derivative of cost in respect to this neuron's activation from past batch
void Neuron::resetNudges()
{
	activationNudgeSum = 0.0;
}

//returns number of input neurons
int Neuron::getInputCount() const
{
	return neuronInputListCount;
}

//returns activation value of neuron
double Neuron::getActivation() const
{
	return activation;
}

//returns weight from this neuron towards a specified input neuron
double Neuron::getWeight(int inputNeuronIndex) const
{
	assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

	return weights[inputNeuronIndex];
}

std::vector<double> Neuron::getWeights() const
{
	std::vector<double> weights;

	for (auto i = 0; i < neuronInputListCount; i++)
	{
		weights.push_back(getWeight(i));
	}

	return weights;
}

//returns bias of this neuron
double Neuron::getBias() const
{
	return bias;
}

//returns the activation type of the neuron
std::string Neuron::getNeuronType()
{
	return getInputCount() == 0 ? "Input" : "Linear";
}


//Set error of neurons with activations directly used to calculate cost dC/da
void NeuralLayer::setError(double costArray[])
{
	if (costArray != nullptr)
	{
		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
			neurons[i].setError(costArray[i]);
	}
}

//nudge input layer activations with appropriate derivatives of cost function dC/da * da/di
void NeuralLayer::injectErrorBackwards()
{
	for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		neurons[i].injectInputRespectiveCostDerivation();
}

//apply learned weights and bias updates
void NeuralLayer::updateParameters(int batchSize, double learningRate, double momentumRetention)
{
	for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
	{
		neurons[i].updateWeights(batchSize, learningRate, momentumRetention);

		neurons[i].updateBias(batchSize, learningRate, momentumRetention);
	}
}

//clears all stored nudges to neuron parameters
void NeuralLayer::clearNudges()
{
	for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		neurons[i].resetNudges();
}

//default constructor for layer class
NeuralLayer::NeuralLayer()
{
	neurons = nullptr;
	neuronArrayLength = 0;
	neuronArrayWidth = 0;
	previousLayer = nullptr;
}

//constructor called for input layers
NeuralLayer::NeuralLayer(int inputLength, int inputWidth) : neuronArrayLength(inputLength), neuronArrayWidth(inputWidth), previousLayer(nullptr)
{
	neurons = new Neuron[inputLength * inputWidth];
	if (neurons == nullptr) throw std::bad_alloc();

	for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
	{
		neurons[i] = Neuron();
	}
}

//constructor called for hidden layers during network creation, with optional momentum parameter
NeuralLayer::NeuralLayer(int neuronCount, NeuralLayer* inputLayer)
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
		neurons[i] = Neuron(inputNeuronCount, inputNeurons);
	}

}

//constructor called for hidden layers during network loading, with stored weights and bias values passed in
NeuralLayer::NeuralLayer(int neuronCount, NeuralLayer* inputLayer, std::vector<std::vector<double>> weightValues, std::vector<double> biasValues)
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
		neurons[i] = Neuron(inputNeuronCount, inputNeurons, weightValues[i], biasValues[i]);
	}
}

//copy constructor for layers
NeuralLayer::NeuralLayer(const NeuralLayer& original)
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
NeuralLayer& NeuralLayer::operator=(const NeuralLayer& original)
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
NeuralLayer::~NeuralLayer()
{
	delete[] neurons;

	previousLayer = nullptr;
}

//activate all neurons in layer and resets nudges from past learning iteration
void NeuralLayer::propagateForward(double inputValues[])
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
void NeuralLayer::propagateBackward(int batchSize, double learningRate, double momentumRetention, double* costArray)
{
	setError(costArray);

	injectErrorBackwards(); //todo: skip for 2nd layer?

	updateParameters(batchSize, learningRate, momentumRetention);
}

//returns number of neurons contained within a column of the layer
int NeuralLayer::getNeuronArrayLength() const
{
	return neuronArrayLength;
}

//returns number of neurons contained within a row of the layer
int NeuralLayer::getNeuronArrayWidth() const
{
	return neuronArrayWidth;
}

//returns number of neurons contained within layer
int NeuralLayer::getNeuronArrayCount() const
{
	return getNeuronArrayLength() * getNeuronArrayWidth();
}

//returns array of pointers to neurons contained within layer
Neuron* NeuralLayer::getNeurons() const
{
	return neurons;
}

//returns pointer to layer that is feeding into this layer
NeuralLayer* NeuralLayer::getPreviousLayer() const
{
	return previousLayer;
}

std::vector<double> NeuralLayer::getNeuronActivations() const
{
	std::vector<double> neuronActivations;

	for (auto i = 0; i < getNeuronArrayCount(); i++)
	{
		neuronActivations.push_back(getNeurons()[i].getActivation());
	}

	return neuronActivations;
}

std::vector<std::vector<double>> NeuralLayer::getNeuronWeights() const
{
	std::vector<std::vector<double>> neuronWeights;

	for (auto i = 0; i < getNeuronArrayCount(); i++)
	{
		neuronWeights.push_back(getNeurons()[i].getWeights());
	}

	return neuronWeights;
}

std::vector<double> NeuralLayer::getNeuronBiases() const
{
	std::vector<double> neuronBiases;

	for (auto i = 0; i < getNeuronArrayCount(); i++)
	{
		neuronBiases.push_back(getNeurons()[i].getBias());
	}

	return neuronBiases;
}

//returns the activation type of the neurons contained within layer
int NeuralLayer::getNeuralLayerType() const
{
	return 1;
}

//the derivation of the mean-squared-error function in respect to the activation of an output neuron
double derivedMSECost(double targetValue, double estimatedValue, int outputCount)
{
	return (-2.0 / (double)outputCount) * (targetValue - estimatedValue);
}

//flips byte ordering of input integer
unsigned int flipIntegerByteOrdering(int original)
{
	unsigned char firstByte, secondByte, thirdByte, fourthByte;

	firstByte = (0xFF000000 & original) >> 24;
	secondByte = (0x00FF0000 & original) >> 16;
	thirdByte = (0x0000FF00 & original) >> 8;
	fourthByte = 0x000000FF & original;

	return ((unsigned int)fourthByte << 24) | ((unsigned int)thirdByte << 16) | ((unsigned int)secondByte << 8) | ((unsigned int)firstByte << 0);
}

//todo: pass in a path string instead of a bool
//returns vector of all available testing or training labels in the dataset
std::vector<unsigned char> getMNISTLabelVector(bool testing)
{
	std::vector<unsigned char> labels;
	unsigned int magicNumber, labelCount;
	unsigned char currentLabel;

	std::string fullPath = "train-labels.idx1-ubyte";
	if (testing) fullPath = "t10k-labels.idx1-ubyte";

	std::ifstream file(fullPath);
	if (file.is_open())
	{
		file.read((char*)&magicNumber, sizeof(magicNumber));
		file.read((char*)&labelCount, sizeof(labelCount));

		magicNumber = flipIntegerByteOrdering(magicNumber);
		labelCount = flipIntegerByteOrdering(labelCount);

		for (auto i = 0; i < labelCount; i++)
		{
			file.read((char*)&currentLabel, sizeof(currentLabel));
			labels.push_back(currentLabel);
		}
	}

	return labels;
}

//todo: pass in a path string instead of a bool
//returns vector of all available testing or training samples in the dataset
std::vector<std::vector<std::vector<unsigned char>>> getMNISTImageVector(bool testing)
{
	std::vector<std::vector<std::vector<unsigned char>>> images;
	std::vector<std::vector<unsigned char>> columnsOfAnImage;
	std::vector<unsigned char> pixelsOfAColumn;

	std::string fullPath = "train-images.idx3-ubyte";
	if (testing) fullPath = "t10k-images.idx3-ubyte";

	unsigned int magicNumber, numberOfImages, rowsPerImage, columnsPerImage;
	unsigned char currentPixel;

	std::ifstream file(fullPath);

	if (file.is_open())
	{
		file.read((char*)&magicNumber, sizeof(magicNumber));
		file.read((char*)&numberOfImages, sizeof(numberOfImages));
		file.read((char*)&rowsPerImage, sizeof(rowsPerImage));
		file.read((char*)&columnsPerImage, sizeof(columnsPerImage));

		magicNumber = flipIntegerByteOrdering(magicNumber);
		numberOfImages = flipIntegerByteOrdering(numberOfImages);
		rowsPerImage = flipIntegerByteOrdering(rowsPerImage);
		columnsPerImage = flipIntegerByteOrdering(columnsPerImage);

		for (auto i = 0; i < numberOfImages; i++)
		{
			for (auto j = 0; j < rowsPerImage; j++)
			{
				for (auto k = 0; k < columnsPerImage; k++)
				{
					file.read((char*)&currentPixel, sizeof(currentPixel));
					pixelsOfAColumn.push_back(currentPixel);
				}

				columnsOfAnImage.push_back(pixelsOfAColumn);
				pixelsOfAColumn.clear();
			}

			images.push_back(columnsOfAnImage);
			columnsOfAnImage.clear();
		}
	}

	return images;
}


//constructor for creating NeuralNetworks
NeuralNetwork::NeuralNetwork(int layerCount, int inputLength, int inputWidth, int outputCount, int costSelection, layerCreationInfo* layerDetails, hyperParameters learningParameters)
{
	this->layerCount = layerCount;
	this->inputLength = inputLength;
	this->inputWidth = inputWidth;
	this->outputCount = outputCount;
	this->learningParameters = learningParameters;

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
			this->neuralLayers[i] = NeuralLayer(layerDetails[i].neuronCount, &neuralLayers[i - 1]);
			break;
		default:
			this->neuralLayers[i] = NeuralLayer(layerDetails[i].neuronCount, &neuralLayers[i - 1]);
			break;
		}
	}

	layerStates = new layerLoadingInfo[layerCount];
}

//constructor for loading NeuralNetworks
NeuralNetwork::NeuralNetwork(int layerCount, int inputLength, int inputWidth, int outputCount, int costSelection, layerLoadingInfo* layerDetails, hyperParameters learningParameters)
{
	this->layerCount = layerCount;
	this->inputLength = inputLength;
	this->inputWidth = inputWidth;
	this->outputCount = outputCount;
	this->learningParameters = learningParameters;

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
			this->neuralLayers[i] = NeuralLayer(layerDetails[i].neuronCount, &neuralLayers[i - 1], layerDetails[i].weightsOfNeurons, layerDetails[i].biasOfNeurons);
			break;
		default:
			this->neuralLayers[i] = NeuralLayer(layerDetails[i].neuronCount, &neuralLayers[i - 1], layerDetails[i].weightsOfNeurons, layerDetails[i].biasOfNeurons);
			break;
		}
	}

	layerStates = new layerLoadingInfo[layerCount];
}

//returns a vector of the activation values of the final layer of the network
std::vector<double> NeuralNetwork::getOutputs()
{
	return neuralLayers[layerCount - 1].getNeuronActivations();
}

//activates all layers in order from input to output layers
void NeuralNetwork::propagateForwards(double* inputMatrix)
{
	neuralLayers[0].propagateForward(inputMatrix);

	for (auto i = 1; i < layerCount; i++)
	{
		neuralLayers[i].propagateForward();
	}
}

//updates parameters in all layers in order from output to input layers
void NeuralNetwork::propagateBackwards(double* costArray)
{
	neuralLayers[layerCount - 1].propagateBackward(learningParameters.batchSize, learningParameters.learningRate, learningParameters.momentumRetention, costArray);

	for (auto i = layerCount - 2; i > 0; i--)
	{
		neuralLayers[i].propagateBackward(learningParameters.batchSize, learningParameters.learningRate, learningParameters.momentumRetention);
	}
}

//changes number of samples network expects to process before being told to learn
/*void NeuralNetwork::updateBatchSize(int newBatchSize)
{
	batchSize = newBatchSize;
}

//updates magnitude of parameter changes during learning
void NeuralNetwork::updateLearningRate(int newLearningRate)
{
	learningRate = newLearningRate;
}*/

//loads training samples from dataset
void NeuralNetwork::updateTrainingSamples()
{
	trainingSamples = getMNISTImageVector(false);
}

//loads training labels from dataset
void NeuralNetwork::updateTrainingLabels()
{
	trainingLabels = getMNISTLabelVector(false);
}

//loads testing samples from dataset
void NeuralNetwork::updateTestingSamples()
{
	testingSamples = getMNISTImageVector(true);
}

//loads testing labels from dataset
void NeuralNetwork::updateTestingLabels()
{
	testingLabels = getMNISTLabelVector(true);
}

//indicates if dataset training samples and labels have been loaded
bool NeuralNetwork::isReadyForTraining()
{
	if (trainingSamples.size() * trainingLabels.size() == 0)
		return false;
	return true;
}

//indicates if dataset testing samples and labels have been loaded
bool NeuralNetwork::isReadyForTesting()
{
	if (testingSamples.size() * testingLabels.size() == 0)
		return false;
	return true;
}

//returns dataset training samples to use during network training
std::vector<std::vector<std::vector<unsigned char>>> NeuralNetwork::getTrainingSamples()
{
	return trainingSamples;
}

//returns dataset training labels to use during network training
std::vector<unsigned char> NeuralNetwork::getTrainingLabels()
{
	return trainingLabels;
}

//returns dataset testing samples to use during network testing
std::vector<std::vector<std::vector<unsigned char>>> NeuralNetwork::getTestingSamples()
{
	return testingSamples;
}

//returns dataset testing labels to use during network testing
std::vector<unsigned char> NeuralNetwork::getTestingLabels()
{
	return testingLabels;
}

//gives the partial derivative value of the cost function in respect to an output activation
double NeuralNetwork::getOutputRespectiveCost(double targetValue, int outputIndex)
{
	return derivedCostFunction(targetValue, getOutputs()[outputIndex], outputCount);
}

//gives the number of inputs that network accepts
int NeuralNetwork::getInputCount()
{
	return inputLength * inputWidth;
}

//gives the number of outputs that the network produces
int NeuralNetwork::getOutputCount()
{
	return outputCount;
}

//gives the depth of the network
int NeuralNetwork::getLayerCount()
{
	return layerCount;
}

//saves all necessary layer data necessary to recreate the network layers exactly upon loading
void NeuralNetwork::saveLayerStates()
{
	for (auto i = 0; i < getLayerCount(); i++)
	{
		layerStates[i].type = neuralLayers[i].getNeuralLayerType();
		layerStates[i].neuronCount = neuralLayers[i].getNeuronArrayCount();
		layerStates[i].weightsOfNeurons = neuralLayers[i].getNeuronWeights();
		layerStates[i].biasOfNeurons = neuralLayers[i].getNeuronBiases();
	}
}

//gives array of layer state details that may be used to recreate layers
layerLoadingInfo* NeuralNetwork::getLayerStates()
{
	return layerStates;
}

hyperParameters NeuralNetwork::getLearningParameters()
{
	return learningParameters;
}

//saves the entire neural network to an xml, such that all data necessary to rebuild the exact network is stored
void storeNetwork(NeuralNetwork* network, std::string& fileName)
{
	int inputLength, outputLength, networkDepth, optimizationAlgorithm, errorFunction;
	hyperParameters learningParameters;

	//saves network details that are not specific to the layers
	inputLength = network->getInputCount();
	outputLength = network->getOutputCount();
	networkDepth = network->getLayerCount();
	optimizationAlgorithm = 0;
	errorFunction = 0;
	learningParameters = network->getLearningParameters();

	//initializes array of fully defined layer states
	network->saveLayerStates();
	layerLoadingInfo* layerStates = network->getLayerStates();

	//defines property trees and subtrees
	boost::property_tree::ptree networkPropertyTree, layerPropertySubTree, neuronPropertySubTree;

	//adds non-layer network details as children to property tree root
	networkPropertyTree.put("network.inputLength", inputLength);
	networkPropertyTree.put("network.outputLength", outputLength);
	networkPropertyTree.put("network.networkDepth", networkDepth);
	networkPropertyTree.put("network.optimizationAlgorithm", optimizationAlgorithm);
	networkPropertyTree.put("network.errorFunction", errorFunction);

	//stores learning hyperparameters
	networkPropertyTree.put("network.learningRate", learningParameters.learningRate);
	networkPropertyTree.put("network.learningDecay", learningParameters.learningDecay);
	networkPropertyTree.put("network.batchSize", learningParameters.batchSize);
	networkPropertyTree.put("network.epochCount", learningParameters.epochCount);
	networkPropertyTree.put("network.momentumRetention", learningParameters.momentumRetention);
	networkPropertyTree.put("network.dropoutPercent", learningParameters.dropoutPercent);
	networkPropertyTree.put("network.outlierMinError", learningParameters.outlierMinError);
	networkPropertyTree.put("network.earlyStoppingMaxError", learningParameters.earlyStoppingMaxError);

	//defines and inserts layer detail subtrees as children to network ptree's 'layers' member
	for (auto i = 0; i < networkDepth; i++)
	{
		//adds non-neuron layer details as chidlren to property subtree root
		layerPropertySubTree.put("activationType", layerStates[i].type);
		layerPropertySubTree.put("neuronCount", layerStates[i].neuronCount);

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

//saves the entire neural network to an xml, such that all data necessary to rebuild the exact network is stored
NeuralNetwork* loadNetworkPointer(const std::string& fileName)
{
	int inputLength, outputLength, networkDepth, optimizationAlgorithm, errorFunction;
	hyperParameters learningParameters;
	boost::property_tree::ptree networkPropertyTree;
	boost::property_tree::read_xml(fileName, networkPropertyTree);

	//saves network details that are not specific to the layers
	inputLength = networkPropertyTree.get<int>("network.inputLength");
	outputLength = networkPropertyTree.get<int>("network.outputLength");
	networkDepth = networkPropertyTree.get<int>("network.networkDepth");
	optimizationAlgorithm = networkPropertyTree.get<int>("network.optimizationAlgorithm");
	errorFunction = networkPropertyTree.get<int>("network.errorFunction");

	//loads learning hyperparameters
	learningParameters.learningRate = networkPropertyTree.get<double>("network.learningRate");
	learningParameters.learningDecay = networkPropertyTree.get<double>("network.learningDecay");
	learningParameters.batchSize = networkPropertyTree.get<double>("network.batchSize");
	learningParameters.epochCount = networkPropertyTree.get<double>("network.epochCount");
	learningParameters.momentumRetention = networkPropertyTree.get<double>("network.momentumRetention");
	learningParameters.dropoutPercent = networkPropertyTree.get<double>("network.dropoutPercent");
	learningParameters.outlierMinError = networkPropertyTree.get<double>("network.outlierMinError");
	learningParameters.earlyStoppingMaxError = networkPropertyTree.get<double>("network.earlyStoppingMaxError");

	layerLoadingInfo* layerStates = new layerLoadingInfo[networkDepth];

	int i = 0;
	std::vector<double> neuronWeights;

	//defines array of layer details by extracting values from the network property tree
	//BOOST_FOREACH(const boost::property_tree::ptree::value_type &layer, networkPropertyTree.get_child("network.layers"))
	for (const boost::property_tree::ptree::value_type& layer : networkPropertyTree.get_child("network.layers"))
	{
		//defines non-neuron layer state details
		layerStates[i].type = layer.second.get<int>("activationType");
		layerStates[i].neuronCount = layer.second.get<int>("neuronCount");

		//defines neuron state details
		//BOOST_FOREACH(const boost::property_tree::ptree::value_type &neuron, layer.second.get_child("layer.neurons"))
		for (const boost::property_tree::ptree::value_type& neuron : layer.second.get_child("neurons"))
		{
			//define neuron's saved bias parameter
			layerStates[i].biasOfNeurons.push_back(neuron.second.get<double>("bias"));

			//define neuron's saved weight parameters, skipping the first layer's weights to avoid get_child exception
			//BOOST_FOREACH(const boost::property_tree::ptree::value_type & weight, neuron.second.get_child("weights"))
			if (i > 0) for (const boost::property_tree::ptree::value_type& weight : neuron.second.get_child("weights"))
			{
				neuronWeights.push_back(weight.second.get_value<double>());
			}

			//store neuron weight parameter array and clear temporary weight value vector for next iteration
			layerStates[i].weightsOfNeurons.push_back(neuronWeights);
			neuronWeights.clear();
		}

		//proceed to defining the next layer's state
		i++;
	}

	//returns fully-defined neural network... todo: might need to overload = operator for NeuralNetwork
	return new NeuralNetwork(networkDepth, inputLength, 1, outputLength, errorFunction, layerStates, learningParameters);
}

//returns the index of the most positive vector element
int getIndexOfMaxEntry(std::vector<double> Vector)
{
	double maxValue = -DBL_MAX, maxIndex = -1;

	for (auto i = 0; i < Vector.size(); i++)
	{
		if (Vector[i] > maxValue)
		{
			maxIndex = i;
			maxValue = Vector[i];
		}
	}

	return maxIndex;
}

//returns the value of the most positive vector element
int getValueOfMaxEntry(std::vector<double> Vector)
{
	int maxValue = -DBL_MAX, maxIndex = -1;

	for (auto i = 0; i < Vector.size(); i++)
	{
		if (Vector[i] > maxValue)
		{
			maxValue = Vector[i];
		}
	}

	return maxValue;
}

//returns the value of the most negative vector element
int getValueOfMinEntry(std::vector<double> Vector)
{
	int minValue = DBL_MAX, minIndex = -1;

	for (auto i = 0; i < Vector.size(); i++)
	{
		if (Vector[i] < minValue)
		{
			minValue = Vector[i];
		}
	}

	return minValue;
}

//Final print before leaving menus
void exitSelection()
{
	std::cout << std::endl;
	std::cout << "Exiting manager..." << std::endl;
}

//lists main menu options and prompts user to select one
MenuStates mainSelection()
{
	int selection;

	//initial menu state prompt to user
	std::cout << std::endl;
	std::cout << "Welcome to the Main Menu!" << std::endl;
	std::cout << "1) Create Neural Network" << std::endl;
	std::cout << "2) Load Neural Network" << std::endl;
	std::cout << "3) Introduction and Info" << std::endl;
	std::cout << "4) Exit Network Manager" << std::endl;
	std::cout << "Selection: ";
	std::cin >> selection;

	//return next menu state
	switch (selection)
	{
	case 1:
		return MenuStates::Create;
	case 2:
		return MenuStates::Load;
	case 3:
		return MenuStates::Intro;
	case 4:
		return MenuStates::Exit;
	default:
		std::cout << std::endl;
		std::cout << "Invalid entry, try again";
		return MenuStates::Main;
	}
}

//todo: improve this
//prints description of project and provides a high-level guide
MenuStates introSelection()
{
	int selection;

	std::cout << std::endl;
	std::cout << "Introduction:" << std::endl;
	std::cout << "Welcome to NeuralNetArchitect! In this pre-alpha console application you can create your own linear ";
	std::cout << "neural network with full model structure and optimization algorithm customizability. Currently, only the ";
	std::cout << "MSE cost function and linear neuron activation functions are available. Datasets can manually input into ";
	std::cout << "the network and learning can only be achieved through the editing of the main method. The menu is a ";
	std::cout << "work in progress:)" << std::endl;

	std::cout << "Type any integer to exit: ";
	std::cin >> selection;

	return MenuStates::Main;
}

//prompts user through creation of a neural network
MenuStates createSelection(NeuralNetwork** network)
{
	int numberOfLayers, inputLength, inputWidth, outputCount, costSelection;
	hyperParameters learningParameters;

	//define input length
	std::cout << std::endl;
	std::cout << "Creation:" << std::endl;
	std::cout << "What is the length of inputs that this neural network will accept? ";
	std::cin >> inputLength;
	std::cout << std::endl;

	//define input width
	//std::cout << "What is the width of inputs that this neural network will accept? ";
	//std::cin >> inputWidth;
	inputWidth = 1;
	//std::cout << std::endl;

	//define output length
	std::cout << "What is the number of outputs that this neural network will produce? ";
	std::cin >> outputCount;
	std::cout << std::endl;

	//define network depth
	std::cout << "How many layers will this neural network contain? ";
	std::cin >> numberOfLayers;
	layerCreationInfo* layerDetails = new layerCreationInfo[numberOfLayers];
	std::cout << std::endl;

	//define cost function that will calculate network's error upon calculating an output
	std::cout << "Which cost function should be used to calculate error? ";
	std::cin >> costSelection;
	costSelection = 1;
	std::cout << std::endl;

	//begin defining hyperparameters
	//define learning rate hyperparameter, the percent of the current learning step error gradient that will update learned parameters
	std::cout << "What is the learning rate of this network? ";
	std::cin >> learningParameters.learningRate;
	learningParameters.learningRate = 0.0000001;
	std::cout << std::endl;

	//define learning decay, the gradual decrease in learning rate of the network after each batch
	std::cout << "What is the learning decay of this network? ";
	std::cin >> learningParameters.learningDecay;
	learningParameters.learningDecay = 0.0;
	std::cout << std::endl;

	//define batch size hyperparameter, the number of samples that will be processed before learning takes place
	std::cout << "What is the batch size that this network will train on? ";
	std::cin >> learningParameters.batchSize;
	learningParameters.batchSize = 1;
	std::cout << std::endl;

	//define epoch count hyperparameter, the number of times the network will train on the data set
	std::cout << "What are the total epochs that this network will train on? ";
	std::cin >> learningParameters.epochCount;
	learningParameters.epochCount = 1;
	std::cout << std::endl;

	//define momentum retention, how much the direction of learning from last batch will influence the learning of this batch
	std::cout << "What is the momentum retention of each training batch? ";
	std::cin >> learningParameters.momentumRetention;
	learningParameters.momentumRetention = 0.0;
	std::cout << std::endl;

	//define percent dropout hyperparameter, the percent of neurons that will be forced inactive to improve generalization
	std::cout << "What is the percent neuron dropout of the network in each batch/epoch? ";
	std::cin >> learningParameters.dropoutPercent;
	learningParameters.dropoutPercent = 0.0;
	std::cout << std::endl;

	//define minimum network error needed to exclude a training image from being learned on
	std::cout << "What is the minimun network error needed to prevent learning on a sample? ";
	std::cin >> learningParameters.outlierMinError;
	learningParameters.outlierMinError = 1.0;
	std::cout << std::endl;

	//define early stopping criteria, maximum percent error that a neural network must achieve on a 'streak' to conclude training
	std::cout << "What is the average maximum percent error that will conclude training? ";
	std::cin >> learningParameters.earlyStoppingMaxError;
	learningParameters.earlyStoppingMaxError = 0.0;
	std::cout << std::endl;

	//initialize first (input) layer
	layerDetails[0].type = 1;
	layerDetails[0].neuronCount = inputLength * inputWidth;

	//define each layer
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

	}

	//create network and point to intialized NeuralNetwork
	*network = new NeuralNetwork(numberOfLayers, inputLength, inputWidth, outputCount, costSelection, layerDetails, learningParameters);

	//return next menu state
	return MenuStates::Manage;
}

//asks user for path of file to load fully-defined neural network from
MenuStates loadSelection(NeuralNetwork** network)
{
	std::string xmlName;

	//acquire name of file to load and initialize NeuralNetwork from
	std::cout << std::endl;
	std::cout << "Loading:" << std::endl;
	std::cout << "Enter XML file name to load from: ";
	std::cin >> xmlName;

	//load network by intializing and pointing to it
	*network = loadNetworkPointer(xmlName);

	//return next menu state
	return MenuStates::Manage;
}

//lists manager options and prompts user to select one
MenuStates manageSelection()
{
	int selection;

	//initial menu state prompt to user
	std::cout << std::endl;
	std::cout << "Manage:" << std::endl;
	std::cout << "1) Select DataSets" << std::endl;
	std::cout << "2) Run Training" << std::endl;
	std::cout << "3) Run Testing" << std::endl;
	std::cout << "4) Save Solution" << std::endl;
	std::cout << "5) Help" << std::endl;
	std::cout << "6) Back" << std::endl;
	std::cout << "Selection: ";
	std::cin >> selection;

	//return next menu state
	switch (selection)
	{
	case 1:
		return MenuStates::Dataset;
	case 2:
		return MenuStates::Training;
	case 3:
		return MenuStates::Testing;
	case 4:
		return MenuStates::Save;
	case 5:
		return MenuStates::Help;
	case 6:
		return MenuStates::Main;
	default:
		std::cout << std::endl;
		std::cout << "Invalid entry, try again";
		return MenuStates::Manage;
	}
}

//asks user for datatset label and sample files and loads them into vectors
MenuStates datasetSelection(NeuralNetwork* network)
{
	std::string trainingImageFilePath, trainingLabelFilePath, testingImageFilePath, testingLabelFilePath;

	//updateTestingSamples

	std::cout << std::endl;
	std::cout << "Dataset:" << std::endl;
	std::cout << "Training set image file path: ";
	std::cin >> trainingImageFilePath;
	std::cout << "Training set label file path: ";
	std::cin >> trainingLabelFilePath;
	std::cout << "Testing set image file path: ";
	std::cin >> testingImageFilePath;
	std::cout << "Testing set label file path: ";
	std::cin >> testingLabelFilePath;

	network->updateTrainingSamples();
	network->updateTrainingLabels();
	network->updateTestingSamples();
	network->updateTestingLabels();

	return MenuStates::Manage;
}

//asks user to define higher-level hyperparameters and commences training
MenuStates trainingSelection(NeuralNetwork* network)
{
	int batchSize, learningRate;
	int minOutputValue, maxOutputValue;
	int selection, answer, correctDeterminations = 0;
	double* inputGrid = nullptr;
	double* errorVector = nullptr;
	bool errorEncountered = false;

	std::vector<std::vector<std::vector<unsigned char>>> trainingSamples = network->getTrainingSamples();
	std::vector<unsigned char> trainingLabels = network->getTrainingLabels();

	if (!network->isReadyForTraining())
	{
		std::cout << "Testing data not yet loaded" << std::endl;
		errorEncountered = true;
	}

	if (network->getInputCount() != trainingSamples[0].size() * trainingSamples[0][0].size())
	{
		std::cout << "Mismatch between dataset input samples and network input count" << std::endl;
		errorEncountered = true;
	}

	if (network->getOutputCount() != 10)
	{
		std::cout << "Mismatch between dataset label type count and network output count" << std::endl;
		errorEncountered = true;
	}

	if (!errorEncountered)
	{
		inputGrid = new double[network->getInputCount()];
		errorVector = new double[network->getOutputCount()];

		//for each image in the set
		for (auto i = 0; i < trainingSamples.size(); i++)
		{	//for each column in an image
			for (auto j = 0; j < trainingSamples[0].size(); j++)
			{	//for each pixel in a column
				for (auto k = 0; k < trainingSamples[0][0].size(); k++)
				{
					//load a pixel
					inputGrid[j * trainingSamples[0].size() + k] = trainingSamples[i][j][k];
				}
			}

			//propagate network forwards to calculate outputs from inputs
			network->propagateForwards(inputGrid);

			//get index of entry that scored the highest, from 0 to 9
			answer = getIndexOfMaxEntry(network->getOutputs());

			if (answer == (int)trainingLabels[i])
			{
				correctDeterminations++;
			}

			if (i % 100 == 0 && i > 0)
			{
				std::cout << "Current score: " << (double)correctDeterminations / (double)i << std::endl;
				std::cout << "answer: " << answer << "\t" << "correct: " << (int)trainingLabels[i] << std::endl;
				std::cout << std::endl;
			}

			minOutputValue = getValueOfMinEntry(network->getOutputs());
			maxOutputValue = getValueOfMaxEntry(network->getOutputs());

			//calculate error vector
			for (auto i = 0; i < network->getOutputCount(); i++)
			{//todo: Cost function would go here, default to partial dC/da of MSE Cost Function
				if (i == (int)trainingLabels[i]) errorVector[i] = network->getOutputRespectiveCost(maxOutputValue, i);
				else errorVector[i] = network->getOutputRespectiveCost(minOutputValue, i);
			}

			network->propagateBackwards(errorVector);
		}

		std::cout << "Final score: " << (double)correctDeterminations / (double)trainingLabels.size() << std::endl;
	}

	std::cout << "Type 0 to exit:" << std::endl;
	std::cin >> selection;

	return MenuStates::Manage;
}

//completes testing of neural network with current learned-parameter values
MenuStates testingSelection(NeuralNetwork* network)
{
	int selection, answer, correctDeterminations = 0;
	double* inputGrid = nullptr;
	bool errorEncountered = false;

	std::vector<std::vector<std::vector<unsigned char>>> testingSamples = network->getTestingSamples();
	std::vector<unsigned char> testingLabels = network->getTestingLabels();

	std::cout << std::endl;
	std::cout << "Testing:" << std::endl;

	if (!network->isReadyForTesting())
	{
		std::cout << "Testing data not yet loaded" << std::endl;
		errorEncountered = true;
	}

	if (network->getInputCount() != testingSamples[0].size() * testingSamples[0][0].size())
	{
		std::cout << "Mismatch between dataset input samples and network input count" << std::endl;
		errorEncountered = true;
	}

	if (network->getOutputCount() != 10)
	{
		std::cout << "Mismatch between dataset label type count and network output count" << std::endl;
		errorEncountered = true;
	}

	if (!errorEncountered)
	{
		inputGrid = new double[network->getInputCount()];

		//for each image in the set
		for (auto i = 0; i < testingSamples.size(); i++)
		{	//for each column in an image
			for (auto j = 0; j < testingSamples[0].size(); j++)
			{	//for each pixel in a column
				for (auto k = 0; k < testingSamples[0][0].size(); k++)
				{
					//load a pixel
					inputGrid[j * testingSamples[0].size() + k] = testingSamples[i][j][k];
				}
			}

			//propagate network forwards to calculate outputs from inputs
			network->propagateForwards(inputGrid);

			//get index of entry that scored the highest, from 0 to 9
			answer = getIndexOfMaxEntry(network->getOutputs());

			if (answer == (int)testingLabels[i])
			{
				correctDeterminations++;
			}

			if (i % 100 == 0 && i > 0)
			{
				std::cout << "Current score: " << (double)correctDeterminations / (double)i << std::endl;
			}


		}

		std::cout << "Final score: " << (double)correctDeterminations / (double)testingLabels.size() << std::endl;
	}

	std::cout << "Type 0 to exit:" << std::endl;
	std::cin >> selection;

	return MenuStates::Manage;
}

//asks user for path of file to store fully-defined neural network in
MenuStates saveSelection(NeuralNetwork* network)
{
	std::string xmlFileName;

	std::cout << std::endl;
	std::cout << "Save:" << std::endl;
	std::cout << "Enter name of file to save network as: ";
	std::cin >> xmlFileName;
	storeNetwork(network, xmlFileName);
	return MenuStates::Manage;
}

//prints detailed instructions and explanation on the customization options
MenuStates helpSelection()
{
	int selection;

	std::cout << std::endl;
	std::cout << "Help:" << std::endl;
	std::cout << "Help of 'manage' options not yet written, dead end on menu" << std::endl;
	std::cout << "Type any integer to exit: ";
	std::cin >> selection;
	return MenuStates::Manage;
}

//indicates error if an invalid menu state is somehow reached
MenuStates defaultSelection()
{
	std::cout << std::endl;
	std::cout << "If you got here, it's a bug. Returning to Main Menu..." << std::endl;
	return MenuStates::Main;
}

//contains full fuctionality of neural network manager Finite State Menu
void manageNeuralNetwork()
{
	NeuralNetwork* network = nullptr;
	MenuStates menuFSMState = MenuStates::Main;

	while (menuFSMState != MenuStates::Exit)
	{
		switch (menuFSMState)
		{
		case MenuStates::Exit:
			exitSelection();
			return;

		case MenuStates::Main:
			menuFSMState = mainSelection();
			break;

		case MenuStates::Intro:
			menuFSMState = introSelection();
			break;

		case MenuStates::Create:
			menuFSMState = createSelection(&network);
			break;

		case MenuStates::Load:
			menuFSMState = loadSelection(&network);
			break;

		case MenuStates::Manage:
			menuFSMState = manageSelection();
			break;

		case MenuStates::Dataset:
			menuFSMState = datasetSelection(network);
			break;

		case MenuStates::Training:
			menuFSMState = trainingSelection(network);
			break;

		case MenuStates::Testing:
			menuFSMState = testingSelection(network);
			break;

		case MenuStates::Save:
			menuFSMState = saveSelection(network);
			break;

		case MenuStates::Help:
			menuFSMState = helpSelection();
			break;

		default:
			menuFSMState = defaultSelection();
			break;
		}
	}
}
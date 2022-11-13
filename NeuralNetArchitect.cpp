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

//Computes neuron's internal sumproduct(weights, inputs) + bias
double Neuron::getActivationFunctionInput() const
{
	double sumOfProducts = 0;
	for (auto i = 0; i < neuronInputListCount; i++)
	{
		sumOfProducts += weights[i] * inputNeurons[i]->getActivation();
	}

	return sumOfProducts + bias;
}

//returns the current value for derivative of cost function in respect to this neuron's activation
double Neuron::getActivationNudgeSum() const
{
	return activationNudgeSum;
}

//Calculates partial derivative of cost function in respect to indexed input neuron activation: dC/da * da/di = dC/di
double Neuron::getActivationRespectiveDerivation(const int inputNeuronIndex) const
{
	assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

	return getActivationNudgeSum() * weights[inputNeuronIndex];
}

//Calculates partial derivative of cost function in respect to indexed weight: dC/da * da/dw = dC/dw
double Neuron::getWeightRespectiveDerivation(const int inputNeuronIndex) const
{
	assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

	return getActivationNudgeSum() * inputNeurons[inputNeuronIndex]->getActivation();
}

//Calculates partial derivative of cost function in respect to indexed input neuron activation: dC/da * da/db = dC/db
double Neuron::getBiasRespectiveDerivation() const
{
	assert(neuronInputListCount >= 0);

	return getActivationNudgeSum() * 1.0;
}

//Adds desired change in activation value that's projected to reduce batch training error, dC/da = completeSum(dC/do * do/da)
void Neuron::nudgeActivation(double nudge)
{
	activationNudgeSum += nudge;
}

//constructor called for input neurons of activation value directly determined by dataset samples
Neuron::Neuron() : weights(nullptr), weightsMomentum(nullptr)
{
	this->neuronInputListCount = 0;

	bias = biasMomentum = 0.0;

	activation = activationNudgeSum = 0.0;
}

//constructor called for hidden neurons during network creation
Neuron::Neuron(int neuronInputListCount, std::vector<Neuron*> inputNeurons)
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

	//Sets weight residual-momentum values to 0
	weightsMomentum = new double[neuronInputListCount]();
	if (weightsMomentum == nullptr) throw std::bad_alloc();

	//Sets bias and bias residual-momentum to 0
	bias = biasMomentum = 0.0;

	//Sets activation and partial derivative dCost/dActivation to 0
	activation = activationNudgeSum = 0.0;
}

//constructor called for hidden neurons during network loading, with stored weights and bias values passed in
Neuron::Neuron(int neuronInputListCount, std::vector<Neuron*> inputNeurons, std::vector<double> weightValues, double biasValue)
{
	this->neuronInputListCount = neuronInputListCount;
	this->inputNeurons = inputNeurons;

	//Initializes weights using loaded values
	weights = new double[neuronInputListCount];
	if (weights == nullptr) throw std::bad_alloc();
	for (auto i = 0; i < neuronInputListCount; i++)
		weights[i] = weightValues[i];

	//Sets weight residual-momentum values to loaded values
	weightsMomentum = new double[neuronInputListCount]();
	if (weightsMomentum == nullptr) throw std::bad_alloc();

	//Sets bias to loaded value and bias residual-momentum to 0
	bias = biasValue;
	biasMomentum = 0.0;

	//Sets activation and partial derivative dCost/dActivation to 0
	activation = activationNudgeSum = 0.0;
}

//copy constructor for neurons, resulting in identical deep copies - future use?
Neuron::Neuron(const Neuron& original)
{
	//Copies non-array values
	neuronInputListCount = original.neuronInputListCount;
	inputNeurons = original.inputNeurons;
	activation = original.activation;
	activationNudgeSum = original.activationNudgeSum;
	bias = original.bias;
	biasMomentum = original.biasMomentum;

	//Copies weight values from original neuron
	weights = new double[neuronInputListCount];
	if (weights == nullptr) throw std::bad_alloc();
	for (auto i = 0; i < neuronInputListCount; i++)
		weights[i] = original.weights[i];

	//Copies weight residual-momentums from original neuron
	weightsMomentum = new double[neuronInputListCount];
	if (weightsMomentum == nullptr) throw std::bad_alloc();
	for (auto i = 0; i < neuronInputListCount; i++)
		weightsMomentum[i] = original.weightsMomentum[i];
}

//operator = overloading for readable assignments, resulting in identical deep copies
Neuron& Neuron::operator=(const Neuron& original)
{
	//Copies non-array values
	neuronInputListCount = original.neuronInputListCount;
	inputNeurons = original.inputNeurons;
	activation = original.activation;
	activationNudgeSum = original.activationNudgeSum;
	bias = original.bias;
	biasMomentum = original.biasMomentum;

	//Copies weight values from original neuron
	weights = new double[neuronInputListCount];
	if (weights == nullptr) throw std::bad_alloc();
	for (auto i = 0; i < neuronInputListCount; i++)
		weights[i] = original.weights[i];

	//Copies weight residual-momentums from original neuron
	weightsMomentum = new double[neuronInputListCount];
	if (weightsMomentum == nullptr) throw std::bad_alloc();
	for (auto i = 0; i < neuronInputListCount; i++)
		weightsMomentum[i] = original.weightsMomentum[i];

	//returns address of newly-created neuron to allow chain assignments
	return *this;
}

//custom destructor for neurons to free array memory and unlink from input neurons
Neuron::~Neuron()
{
	for (std::vector<Neuron*>::iterator it = inputNeurons.begin(); it != inputNeurons.end(); ++it)
	{
		*it = nullptr;
	}

	delete[] weights;
	delete[] weightsMomentum;
}

//Defines empty exterior activation function of neuron, leaving only a linear sumOfProducts(weights,inputActivations) + bias
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

//Injects error dC/da into this neuron, called for output neurons that are directly used to calculate cost
void Neuron::setError(double cost)
{
	activationNudgeSum = cost;
}

//Injects corresponding error into input neurons due to activation, dC/di = dC/dh * dh/di for each input activation i
void Neuron::injectInputRespectiveCostDerivation() const
{
	for (auto i = 0; i < neuronInputListCount; i++)
	{
		inputNeurons[i]->nudgeActivation(getActivationRespectiveDerivation(i));
	}
}

//Applies change to weights that is projected to reduce cost for past batch, defines weight-update portion of a learning step
void Neuron::updateWeights(int batchSize, double learningRate, double momentumRetention)
{
	//Calculates weight residual-momentums for next learning step based on hyperparameter
	for (auto i = 0; i < neuronInputListCount; i++)
	{
		//Uses reserved activationNudges to scale change proportionally to a neuron's effect on network error
		weightsMomentum[i] = momentumRetention * weightsMomentum[i] - (getWeightRespectiveDerivation(i) / batchSize) * learningRate;
		weights[i] += weightsMomentum[i];
	}
}

//Applies change to bias that is projected to reduce cost for past batch, defines bias-update portion of a learning step
void Neuron::updateBias(int batchSize, double learningRate, double momentumRetention)
{
	//Calculates weight residual-momentums for next learning step based on hyperparameter, scales to activationNudgeSum
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

//returns current activation value of neuron
double Neuron::getActivation() const
{
	return activation;
}

//returns weight from this neuron that scales a specified input neuron's activation
double Neuron::getWeight(int inputNeuronIndex) const
{
	assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

	return weights[inputNeuronIndex];
}

//returns current learned weight values
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

//returns the activation type of the neuron -unused?
std::string Neuron::getNeuronType()
{
	return getInputCount() == 0 ? "Input" : "Linear";
}


//constructor called for hidden ReLU neurons during network creation
ReLUNeuron::ReLUNeuron(int neuronInputListCount, std::vector<Neuron*> inputNeurons)
	: Neuron(neuronInputListCount, inputNeurons) {
	std::cout << "Created" << std::endl;
}

//constructor called for hidden ReLU neurons during network loading, with previously-stored parameter values passed in
ReLUNeuron::ReLUNeuron(int neuronInputListCount, std::vector<Neuron*> inputNeurons, std::vector<double> weightValues, double biasValue)
	: Neuron(neuronInputListCount, inputNeurons, weightValues, biasValue) {}

//Calculates partial derivative of cost function in respect to indexed input neuron activation: dC/da * da/di = dC/di
double ReLUNeuron::getActivationRespectiveDerivation(const int inputNeuronIndex) const
{
	assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

	return (getActivation() > 0) ? getActivationNudgeSum() * weights[inputNeuronIndex] : 0;
}

//Calculates partial derivative of cost function in respect to indexed weight: dC/da * da/dw = dC/dw
double ReLUNeuron::getWeightRespectiveDerivation(const int inputNeuronIndex) const
{
	assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

	return (getActivation() > 0) ? getActivationNudgeSum() * inputNeurons[inputNeuronIndex]->getActivation() : 0;
}

//Calculates partial derivative of cost function in respect to indexed input neuron activation: dC/da * da/db = dC/db
double ReLUNeuron::getBiasRespectiveDerivation() const
{
	assert(neuronInputListCount >= 0);

	return (getActivation() > 0) ? getActivationNudgeSum() * 1.0 : 0;
}

//Defines ReLU exterior activation function of neuron, ReLU(sumOfProducts(weights,inputActivations) + bias)
void ReLUNeuron::activate(const double input)
{
	if (neuronInputListCount > 0)
	{
		activation = (getActivationFunctionInput() > 0) ? getActivationFunctionInput() : 0;
	}
	else
	{
		activation = input;
	}
}

//returns the activation type of the neuron -unused?
std::string ReLUNeuron::getNeuronType()
{
	return "ReLU";
}


//constructor called for hidden Sigmoid neurons during network creation
SigmoidNeuron::SigmoidNeuron(int neuronInputListCount, std::vector<Neuron*> inputNeurons)
	: Neuron(neuronInputListCount, inputNeurons) {}

//constructor called for hidden Sigmoid neurons during network loading, with previously-stored parameter values passed in
SigmoidNeuron::SigmoidNeuron(int neuronInputListCount, std::vector<Neuron*> inputNeurons, std::vector<double> weightValues, double biasValue)
	: Neuron(neuronInputListCount, inputNeurons, weightValues, biasValue) {}

//Calculates partial derivative of cost function in respect to indexed input neuron activation: dC/da * da/di = dC/di
double SigmoidNeuron::getActivationRespectiveDerivation(const int inputNeuronIndex) const
{
	assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

	return getActivationNudgeSum() * getActivation() * (1 - getActivation()) * weights[inputNeuronIndex];
}

//Calculates partial derivative of cost function in respect to indexed weight: dC/da * da/dw = dC/dw
double SigmoidNeuron::getWeightRespectiveDerivation(const int inputNeuronIndex) const
{
	assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

	return getActivationNudgeSum() * getActivation() * (1 - getActivation()) * inputNeurons[inputNeuronIndex]->getActivation();
}

//Calculates partial derivative of cost function in respect to indexed input neuron activation: dC/da * da/db = dC/db
double SigmoidNeuron::getBiasRespectiveDerivation() const
{

	assert(neuronInputListCount >= 0);

	return getActivationNudgeSum() * getActivation() * (1 - getActivation()) * 1.0;
}

//Defines ReLU exterior activation function of neuron, ReLU(sumOfProducts(weights,inputActivations) + bias)
void SigmoidNeuron::activate(const double input)
{
	if (neuronInputListCount > 0)
	{
		activation = 1 / (1 + exp(-1 * (getActivationFunctionInput())));
	}
	else
	{
		activation = input;
	}
}

//returns the activation type of the neuron -unused?
std::string SigmoidNeuron::getNeuronType()
{
	return "Sigmoid";
}

//IN PROGRESS SECTION START

//derive limit definition of Dirac Delta
//complete functions
//constructor called for hidden Sigmoid neurons during network creation
BinaryNeuron::BinaryNeuron(int neuronInputListCount, std::vector<Neuron*> inputNeurons)
	: Neuron(neuronInputListCount, inputNeurons) {}

//constructor called for hidden Sigmoid neurons during network loading, with previously-stored parameter values passed in
BinaryNeuron::BinaryNeuron(int neuronInputListCount, std::vector<Neuron*> inputNeurons, std::vector<double> weightValues, double biasValue)
	: Neuron(neuronInputListCount, inputNeurons, weightValues, biasValue) {}

//Calculates partial derivative of cost function in respect to indexed input neuron activation: dC/da * da/di = dC/di
double BinaryNeuron::getActivationRespectiveDerivation(const int inputNeuronIndex) const
{
	assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

	return getActivationNudgeSum() * exp(-1 * getActivationFunctionInput() * getActivationFunctionInput()) * weights[inputNeuronIndex];
}

//Calculates partial derivative of cost function in respect to indexed weight: dC/da * da/dw = dC/dw
double BinaryNeuron::getWeightRespectiveDerivation(const int inputNeuronIndex) const
{
	assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

	return getActivationNudgeSum() * exp(-1 * getActivationFunctionInput() * getActivationFunctionInput()) * inputNeurons[inputNeuronIndex]->getActivation();
}

//Calculates partial derivative of cost function in respect to indexed input neuron activation: dC/da * da/db = dC/db
double BinaryNeuron::getBiasRespectiveDerivation() const
{

	assert(neuronInputListCount >= 0);

	return getActivationNudgeSum() * exp(-1 * getActivationFunctionInput() * getActivationFunctionInput()) * 1.0;
}

//Defines ReLU exterior activation function of neuron, ReLU(sumOfProducts(weights,inputActivations) + bias)
void BinaryNeuron::activate(const double input)
{
	if (neuronInputListCount > 0)
	{
		activation = (getActivationFunctionInput() > 0 ) ? true : false;
	}
	else
	{
		activation = input;
	}
}

//returns the activation type of the neuron -unused?
std::string BinaryNeuron::getNeuronType()
{
	return "Binary";
}


//review Softmax combination solution
//modify constructors to use weight 1 for all connections
//finish functions
//constructor called for hidden Sigmoid neurons during network creation
ExponentialNeuron::ExponentialNeuron(int neuronInputListCount, std::vector<Neuron*> inputNeurons)
	: Neuron(neuronInputListCount, inputNeurons) {}

//constructor called for hidden Sigmoid neurons during network loading, with previously-stored parameter values passed in
ExponentialNeuron::ExponentialNeuron(int neuronInputListCount, std::vector<Neuron*> inputNeurons, std::vector<double> weightValues, double biasValue)
	: Neuron(neuronInputListCount, inputNeurons, weightValues, biasValue) {}

//Calculates partial derivative of cost function in respect to indexed input neuron activation: dC/da * da/di = dC/di
double ExponentialNeuron::getActivationRespectiveDerivation(const int inputNeuronIndex) const
{
	assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

	return getActivationNudgeSum() * getActivation() * weights[inputNeuronIndex];
}

//Calculates partial derivative of cost function in respect to indexed weight: dC/da * da/dw = dC/dw
double ExponentialNeuron::getWeightRespectiveDerivation(const int inputNeuronIndex) const
{
	assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

	return getActivationNudgeSum() * getActivation() * inputNeurons[inputNeuronIndex]->getActivation();
}

//Calculates partial derivative of cost function in respect to indexed input neuron activation: dC/da * da/db = dC/db
double ExponentialNeuron::getBiasRespectiveDerivation() const
{

	assert(neuronInputListCount >= 0);

	return getActivationNudgeSum() * getActivation() * 1.0;
}

//Defines ReLU exterior activation function of neuron, ReLU(sumOfProducts(weights,inputActivations) + bias)
void ExponentialNeuron::activate(const double input)
{
	if (neuronInputListCount > 0)
	{
		activation = exp( getActivationFunctionInput() );
	}
	else
	{
		activation = input;
	}
}

//returns the activation type of the neuron -unused?
std::string ExponentialNeuron::getNeuronType()
{
	return "Exponential";
}


//finish functions
//constructor called for hidden Sigmoid neurons during network creation
SoftmaxNeuron::SoftmaxNeuron(int neuronInputListCount, std::vector<Neuron*> inputNeurons, int numeratorIndex)
	: Neuron(neuronInputListCount, inputNeurons) 
{
	for (auto i = 0; i < neuronInputListCount; i++)
	{
		weights[i] = 1.0;
	}
}

//constructor called for hidden Sigmoid neurons during network loading, with previously-stored parameter values passed in
SoftmaxNeuron::SoftmaxNeuron(int neuronInputListCount, std::vector<Neuron*> inputNeurons, std::vector<double> weightValues, double biasValue, int numeratorIndex)
	: Neuron(neuronInputListCount, inputNeurons, weightValues, biasValue) {}

//Calculates partial derivative of cost function in respect to indexed input neuron activation: dC/da * da/di = dC/di
double SoftmaxNeuron::getActivationRespectiveDerivation(const int inputNeuronIndex) const
{
	assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

	return (inputNeuronIndex == numeratorInputIndex) ? getNumeratorRespectiveDerivation() : getDenominatorRespectiveDerivation(inputNeuronIndex);
}

//Calculates partial derivative of cost function in respect to indexed weight: dC/da * da/dw = dC/dw
double SoftmaxNeuron::getWeightRespectiveDerivation(const int inputNeuronIndex) const
{
	assert(inputNeuronIndex < neuronInputListCount&& inputNeuronIndex >= 0);

	return 0.0;
}

//Calculates partial derivative of cost function in respect to indexed input neuron activation: dC/da * da/db = dC/db
double SoftmaxNeuron::getBiasRespectiveDerivation() const
{

	assert(neuronInputListCount >= 0);

	return 0.0;
}

//Defines ReLU exterior activation function of neuron, ReLU(sumOfProducts(weights,inputActivations) + bias)
void SoftmaxNeuron::activate(const double input)
{
	if (neuronInputListCount > 0)
	{
		activation = getNumerator() / getDenominator();
	}
	else
	{
		activation = input;
	}
}

//returns the activation type of the neuron -unused?
std::string SoftmaxNeuron::getNeuronType()
{
	return "Softmax";
}

void SoftmaxNeuron::updateBias(int batchSize, double learningRate, double momentumRetention)
{
	return;
}

void SoftmaxNeuron::updateWeights(int batchSize, double learningRate, double momentumRetention)
{
	return;
}

double SoftmaxNeuron::getNumerator() const
{
	return inputNeurons[numeratorInputIndex]->getActivation();
}

double SoftmaxNeuron::getDenominator() const
{
	double previousLayerActivationSum = 0;

	for (auto t : inputNeurons)
	{
		previousLayerActivationSum += t->getActivation();
	}

	return previousLayerActivationSum;
}

double SoftmaxNeuron::getNumeratorRespectiveDerivation() const
{
	return getDenominatorRespectiveDerivation() + 1.0 / getDenominator();
}

double SoftmaxNeuron::getDenominatorRespectiveDerivation() const
{
	return -1.0 * getNumerator() / (getDenominator() * getDenominator());
}

//IN PROGRESS SECTION END


//Set error of neurons with activations directly used to calculate cost dC/da
void NeuralLayer::setError(double costArray[])
{
	for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		neurons[i]->setError(costArray[i]);
}

//nudge input layer activations with derivatives of cost function dC/da * da/di
void NeuralLayer::injectErrorBackwards()
{
	for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		neurons[i]->injectInputRespectiveCostDerivation();
}

//apply learned weights and bias updates
void NeuralLayer::updateParameters(int batchSize, double learningRate, double momentumRetention)
{
	for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
	{
		neurons[i]->updateWeights(batchSize, learningRate, momentumRetention);

		neurons[i]->updateBias(batchSize, learningRate, momentumRetention);
	}
}

//clears all stored nudges to neuron parameters
void NeuralLayer::clearNudges()
{
	for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		neurons[i]->resetNudges();
}

//default constructor - todo: remove?
NeuralLayer::NeuralLayer()
{
	neuronArrayLength = 0;
	neuronArrayWidth = 0;
	previousLayer = nullptr;
}

//constructor for initializing input layers
NeuralLayer::NeuralLayer(int inputLength, int inputWidth) : neuronArrayLength(inputLength), neuronArrayWidth(inputWidth), previousLayer(nullptr)
{
	neuronType = 1;

	for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
	{
		neurons.push_back(new Neuron());
	}
}

//constructor for initializing hidden layers during network creation
NeuralLayer::NeuralLayer(int neuronCount, NeuralLayer* inputLayer, int activationType)
{
	neuronArrayLength = neuronCount;
	neuronArrayWidth = 1;
	previousLayer = inputLayer;
	neuronType = activationType;

	int inputNeuronCount = previousLayer->getNeuronArrayCount();
	std::vector<Neuron*> inputNeurons = previousLayer->getNeurons();

	switch (neuronType)
	{
	case 1:
		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons.push_back(new Neuron(inputNeuronCount, inputNeurons));
		}
		break;
	case 2:
		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{//todo: _vfptr of returned ReLUNeuron is correct, but neurons[i] only points to base functions
			neurons.push_back(new ReLUNeuron(inputNeuronCount, inputNeurons));
		}
		break;
	case 3:
		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons.push_back(new SigmoidNeuron(inputNeuronCount, inputNeurons));
		}
		break;
	default:
		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons.push_back(new Neuron(inputNeuronCount, inputNeurons));
		}
		break;
	}
}

//constructor for hidden layers during network loading
NeuralLayer::NeuralLayer(int neuronCount, NeuralLayer* inputLayer, std::vector<std::vector<double>> weightValues, std::vector<double> biasValues, int activationType)
{
	neuronArrayLength = neuronCount;
	neuronArrayWidth = 1;
	previousLayer = inputLayer;
	neuronType = activationType;

	int inputNeuronCount = previousLayer->getNeuronArrayCount();
	std::vector<Neuron*> inputNeurons = previousLayer->getNeurons();

	switch (neuronType)
	{
	case 1:
		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons.push_back(new Neuron(inputNeuronCount, inputNeurons, weightValues[i], biasValues[i]));
		}
		break;
	case 2:
		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons.push_back(new ReLUNeuron(inputNeuronCount, inputNeurons, weightValues[i], biasValues[i]));
		}
		break;
	case 3:
		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons.push_back(new SigmoidNeuron(inputNeuronCount, inputNeurons, weightValues[i], biasValues[i]));
		}
		break;
	default:
		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons.push_back(new Neuron(inputNeuronCount, inputNeurons, weightValues[i], biasValues[i]));
		}
		break;
	}
}

//copy constructor for layer deep copies - todo: accomodate for several Neuron types?
//necessary?
/*/NeuralLayer::NeuralLayer(const NeuralLayer& original)
{
	neuronArrayLength = original.neuronArrayLength;
	neuronArrayWidth = original.neuronArrayWidth;
	previousLayer = original.previousLayer;

	for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
	{
		neurons.push_back(new Neuron(original.neurons[i]));
	}
}*/

//operator = overloading for initializing and returning of object deep copy - todo: accomodate for several Neuron types?
NeuralLayer& NeuralLayer::operator=(const NeuralLayer& original)
{
	neuronArrayLength = original.neuronArrayLength;
	neuronArrayWidth = original.neuronArrayWidth;
	previousLayer = original.previousLayer;
	neurons = original.getNeurons();
	neuronType = original.getNeuralLayerType();

	//for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
	//{
	//	neurons.push_back(new Neuron(*(original.neurons[i])));
	//}

	return (*this);
}

//custom destructor for NeuralLayer objects for complete memory deallocation
NeuralLayer::~NeuralLayer()
{
	neurons.clear();

	previousLayer = nullptr;
}

//activate all neurons in layer and resets activation nudges from previous learning step
void NeuralLayer::propagateForward(double inputValues[])
{
	if (previousLayer == nullptr)
	{
		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons[i]->activate(inputValues[i]);
		}
	}

	else
	{
		for (auto i = 0; i < neuronArrayLength * neuronArrayWidth; i++)
		{
			neurons[i]->activate();
		}
	}

	clearNudges();
}

//inject error to previous layer and apply learned parameter updates to this layer
void NeuralLayer::propagateBackward(int batchSize, double learningRate, double momentumRetention, double* costArray)
{
	setError(costArray);

	injectErrorBackwards(); //todo: skip for 2nd layer?

	updateParameters(batchSize, learningRate, momentumRetention);
}

//inject error to previous layer and apply learned parameter updates to this layer
void NeuralLayer::propagateBackward(int batchSize, double learningRate, double momentumRetention)
{
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

//returns total number of neurons contained within this layer
int NeuralLayer::getNeuronArrayCount() const
{
	return getNeuronArrayLength() * getNeuronArrayWidth();
}

//returns array of pointers to neurons contained within this layer
std::vector<Neuron*> NeuralLayer::getNeurons() const
{
	return neurons;
}

//returns pointer to layer that is feeding into this layer
NeuralLayer* NeuralLayer::getPreviousLayer() const
{
	return previousLayer;
}

//returns vector of this layer's neuron activations
std::vector<double> NeuralLayer::getNeuronActivations() const
{
	std::vector<double> neuronActivations;

	for (auto i = 0; i < getNeuronArrayCount(); i++)
	{
		neuronActivations.push_back(getNeurons()[i]->getActivation());
	}

	return neuronActivations;
}

//returns 2D vector of weight parameter values belonging to neurons of this layer
std::vector<std::vector<double>> NeuralLayer::getNeuronWeights() const
{
	std::vector<std::vector<double>> neuronWeights;

	for (auto i = 0; i < getNeuronArrayCount(); i++)
	{
		neuronWeights.push_back(getNeurons()[i]->getWeights());
	}

	return neuronWeights;
}

//returns vector of bias parameter values belonging to neurons of this layer
std::vector<double> NeuralLayer::getNeuronBiases() const
{
	std::vector<double> neuronBiases;

	for (auto i = 0; i < getNeuronArrayCount(); i++)
	{
		neuronBiases.push_back(getNeurons()[i]->getBias());
	}

	return neuronBiases;
}

//returns the activation type of the neurons contained within this layer
int NeuralLayer::getNeuralLayerType() const
{
	return neuronType;
}

//the derivation of the mean-squared-error function in respect to the activation of an output neuron - todo: rework this for derivations?
double derivedMSECost(double targetValue, double estimatedValue, int outputCount)
{
	return (-2.0 / (double)outputCount) * (targetValue - estimatedValue);
}

//flips byte ordering integer to convert between high and low endian formats
unsigned int flipIntegerByteOrdering(int original)
{
	unsigned char firstByte, secondByte, thirdByte, fourthByte;

	//isolate each of the 4 bytes that make up the integer
	firstByte = (0xFF000000 & original) >> 24;
	secondByte = (0x00FF0000 & original) >> 16;
	thirdByte = (0x0000FF00 & original) >> 8;
	fourthByte = 0x000000FF & original;

	//flip the ordering of the bytes that make up the integer
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
		//read two high-endian integers of the file
		file.read((char*)&magicNumber, sizeof(magicNumber));
		file.read((char*)&labelCount, sizeof(labelCount));

		//flip byte ordering of integers to obtain intended values in low-endian architectures
		magicNumber = flipIntegerByteOrdering(magicNumber);
		labelCount = flipIntegerByteOrdering(labelCount);

		//read and store each label
		for (auto i = 0; i < labelCount; i++)
		{
			file.read((char*)&currentLabel, sizeof(currentLabel));
			labels.push_back(currentLabel);
		}

		file.close();
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
		//read four high-endian integers of the file
		file.read((char*)&magicNumber, sizeof(magicNumber));
		file.read((char*)&numberOfImages, sizeof(numberOfImages));
		file.read((char*)&rowsPerImage, sizeof(rowsPerImage));
		file.read((char*)&columnsPerImage, sizeof(columnsPerImage));

		//flip byte ordering of integers to obtain intended values in low-endian architectures
		magicNumber = flipIntegerByteOrdering(magicNumber);
		numberOfImages = flipIntegerByteOrdering(numberOfImages);
		rowsPerImage = flipIntegerByteOrdering(rowsPerImage);
		columnsPerImage = flipIntegerByteOrdering(columnsPerImage);

		//read and store each pixel of the sample
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

		file.close();
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

	//select which error function the NeuralNetwork will try to minimize
	switch (costSelection)
	{
	case 1:
		this->derivedCostFunction = derivedMSECost;
		break;
	default:
		this->derivedCostFunction = derivedMSECost;
		break;
	}

	//declare array of NeuralLayer pointers
	neuralLayers = new NeuralLayer[layerCount];
	if (neuralLayers == nullptr) throw std::bad_alloc();
	neuralLayers[0] = NeuralLayer(inputLength, inputWidth);

	//initialize NeuralLayers and have array elements point to them
	for (auto i = 1; i < layerCount; i++)
	{
		this->neuralLayers[i] = NeuralLayer(layerDetails[i].neuronCount, &neuralLayers[i - 1], layerDetails[i].activationType);
	}

	//save layer states
	layerStates = new layerLoadingInfo[layerCount];

	offsetNormalizer = 0;
	scalingNormalizer = 1;
}

//constructor for loading NeuralNetworks
NeuralNetwork::NeuralNetwork(int layerCount, int inputLength, int inputWidth, int outputCount, int costSelection, layerLoadingInfo* layerDetails, hyperParameters learningParameters)
{
	this->layerCount = layerCount;
	this->inputLength = inputLength;
	this->inputWidth = inputWidth;
	this->outputCount = outputCount;
	this->learningParameters = learningParameters;

	//select which error function the NeuralNetwork will try to minimize
	switch (costSelection)
	{
	case 1:
		this->derivedCostFunction = derivedMSECost;
		break;
	default:
		this->derivedCostFunction = derivedMSECost;
		break;
	}

	//declare array of NeuralLayer pointers
	neuralLayers = new NeuralLayer[layerCount];
	if (neuralLayers == nullptr) throw std::bad_alloc();
	neuralLayers[0] = NeuralLayer(inputLength, inputWidth);

	//initialize NeuralLayers and have array elements point to them
	for (auto i = 1; i < layerCount; i++)
	{
		this->neuralLayers[i] = NeuralLayer(layerDetails[i].neuronCount, &neuralLayers[i - 1], layerDetails[i].weightsOfNeurons, layerDetails[i].biasOfNeurons, layerDetails[i].activationType);
	}

	//save layer states
	layerStates = new layerLoadingInfo[layerCount];

	offsetNormalizer = 0;
	scalingNormalizer = 1;
}

//returns a vector of the activation values of the final layer of the network
std::vector<double> NeuralNetwork::getOutputs()
{
	return neuralLayers[layerCount - 1].getNeuronActivations();
}

//activates all layers in order from input to output layers
void NeuralNetwork::propagateForwards(double* inputMatrix)
{
	//activates input layer to input values
	neuralLayers[0].propagateForward(inputMatrix);

	//activates remaining layers to their activation function
	for (auto i = 1; i < layerCount; i++)
	{
		neuralLayers[i].propagateForward();
	}
}

//updates parameters in all layers in order from output to input layers
void NeuralNetwork::propagateBackwards(double* costArray)
{
	//informs output layer of the initial error array in first backpropagation call
	neuralLayers[layerCount - 1].propagateBackward(learningParameters.batchSize, learningParameters.learningRate, learningParameters.momentumRetention, costArray);

	//performs backpropagation for all preceeding layers
	for (auto i = layerCount - 2; i > 1; i--)
	{
		neuralLayers[i].propagateBackward(learningParameters.batchSize, learningParameters.learningRate, learningParameters.momentumRetention);
	}

	//todo: maybe add some out of bounds check for this
	neuralLayers[1].updateParameters(learningParameters.batchSize, learningParameters.learningRate, learningParameters.momentumRetention);
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

//Updates offset and scaling normalizers to be mean and standard deviation, to normalize inputs to mean=0 and stdv=1
void NeuralNetwork::updateNormalizers()
{
	double sum = 0, mean = 0, variance = 0, stdv = 0;
	double varianceNumerator = 0;
	int valuesInASample = testingSamples[0].size() * testingSamples[0][0].size();

	//determine the mean of a subset of total samples
	for (auto i = 0; i < testingSamples.size(); i++)
	{
		for (auto j = 0; j < testingSamples[i].size(); j++)
		{
			for (auto k = 0; k < testingSamples[i][j].size(); k++)
			{
				sum += testingSamples[i][j][k];
			}
		}

		//partially determine mean after each sample to prevent overflow
		mean += sum / testingSamples.size();
		sum = 0;
	}

	//determine the variance given the mean
	for (auto i = 0; i < testingSamples.size(); i++)
	{
		for (auto j = 0; j < testingSamples[i].size(); j++)
		{
			for (auto k = 0; k < testingSamples[i][j].size(); k++)
			{
				sum += testingSamples[i][j][k];
			}
		}

		//partially determine variance numerator after each sample to prevent overflow
		varianceNumerator += (mean - sum / valuesInASample) * (mean - sum / valuesInASample);
		sum = 0;
	}

	//finish calculating variance and calculate standard deviation
	variance = varianceNumerator / (testingSamples.size() + 1);
	stdv = std::sqrt(variance);

	offsetNormalizer = mean;
	scalingNormalizer = stdv;

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

void NeuralNetwork::train()
{
	//int batchSize, learningRate; todo:implement
	int minOutputValue, maxOutputValue;
	int selection, answer, correctDeterminations = 0;
	double* inputGrid = nullptr;
	double* errorVector = nullptr;

	if (!isReadyForTraining())
	{
		throw DatasetNotLoadedException("Training dataset not yet loaded");
	}

	//checks if network input dimensions matches dataset sample dimensions
	if (getInputCount() != trainingSamples[0].size() * trainingSamples[0][0].size())
	{
		throw DatasetMismatchException("Mismatch between dataset input samples and network input count");
	}

	//checks if network output length matches cardinality of dataset labels
	//todo: update hard-coded number
	if (getOutputCount() != 10)
	{
		throw DatasetMismatchException("Mismatch between dataset label type count and network output count");
	}

	//perform training om all training samples
	inputGrid = new double[getInputCount()];
	errorVector = new double[getOutputCount()];

	//for each image in the set
	for (auto i = 0; i < trainingSamples.size(); i++)
	{
		//for each column in an image
		for (auto j = 0; j < trainingSamples[0].size(); j++)
		{
			//for each pixel in a column
			for (auto k = 0; k < trainingSamples[0][0].size(); k++)
			{
				//load a pixel
				inputGrid[j * trainingSamples[0].size() + k] = ((double)trainingSamples[i][j][k] - offsetNormalizer) / scalingNormalizer;
			}
		}

		//propagate network forwards to calculate outputs from inputs
		propagateForwards(inputGrid);

		//get index of entry that scored the highest, from 0 to 9
		//todo: sections assumes index number will always match the answer
		answer = getIndexOfMaxEntry(getOutputs());

		//if network guessed the correct answer, count successful attempt
		if (answer == (int)trainingLabels[i])
		{
			correctDeterminations++;
		}

		//periodically displays current network performance
		//todo: possibly make this not hard-coded?
		if (i % 100 == 0 && i > 0)
		{
			std::cout << "Current score: " << (double)correctDeterminations / (double)i << std::endl;
			std::cout << "maxValue: " << getOutputs()[answer] << "\t" << "answer: " << answer << "\t" << "correct: " << (int)trainingLabels[i] << std::endl;
			std::cout << std::endl;
		}

		//$$$work in progress section for training linear neural network
		//todo: get linear training work more probablistically and abstract section for different neuron types
		maxOutputValue = getValueOfMaxEntry(getOutputs());

		//calculate error vector
		for (auto l = 0; l < getOutputCount(); l++)
		{//todo: Cost function would go here, default to partial dC/da of MSE Cost Function
			//todo: Fix this
			if (l == (int)trainingLabels[i] )
			{
				errorVector[l] = getOutputRespectiveCost(5, l);
			}
			else
			{
				errorVector[l] = getOutputRespectiveCost(-5, l);
			}
		}//$$$end of work in progress section

		//perform backpropagation given an error vector
		//todo: make errorVector come from cost function
		//todo: make more cost functions to accomodate step and linear neurons
		propagateBackwards(errorVector);
	}

	delete[] inputGrid;
	delete[] errorVector;

	//display final network training score
	std::cout << "Final score: " << (double)correctDeterminations / (double)trainingLabels.size() << std::endl;
}

void NeuralNetwork::test()
{
	int answer, correctDeterminations = 0;
	double* inputGrid = nullptr;

	std::cout << std::endl;
	std::cout << "Testing:" << std::endl;

	//checks if testing data has previously been loaded
	if (!isReadyForTesting())
	{
		throw DatasetNotLoadedException("Testing dataset not yet loaded");
	}

	//checks if network input dimensions matches dataset sample dimensions
	if (getInputCount() != testingSamples[0].size() * testingSamples[0][0].size())
	{
		throw DatasetMismatchException("Mismatch between dataset input samples and network input count");
	}

	//checks if network output length matches cardinality of dataset labels
	//todo: update hard-coded number
	if (getOutputCount() != 10)
	{
		throw DatasetMismatchException("Mismatch between dataset label type count and network output count");
	}

	//perform testing om all testing samples
	inputGrid = new double[getInputCount()];

	//for each image in the set
	for (auto i = 0; i < testingSamples.size(); i++)
	{	//for each column in an image
		for (auto j = 0; j < testingSamples[0].size(); j++)
		{	//for each pixel in a column
			for (auto k = 0; k < testingSamples[0][0].size(); k++)
			{
				//load a pixel
				inputGrid[j * testingSamples[0].size() + k] = ((double)trainingSamples[i][j][k] - offsetNormalizer) / scalingNormalizer;
			}
		}

		//propagate network forwards to calculate outputs from inputs
		propagateForwards(inputGrid);

		//get index of entry that scored the highest, from 0 to 9
		answer = getIndexOfMaxEntry(getOutputs());

		//if network guessed the correct answer, count successful attempt
		if (answer == (int)testingLabels[i])
		{
			correctDeterminations++;
		}

		//periodically displays current network performance
		//todo: possibly make this not hard-coded?
		if (i % 100 == 0 && i > 0)
		{
			std::cout << "Current score: " << (double)correctDeterminations / (double)i << std::endl;
		}
	}

	delete[] inputGrid;

	//display final network training score
	std::cout << "Final score: " << (double)correctDeterminations / (double)testingLabels.size() << std::endl;
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
		layerStates[i].activationType = neuralLayers[i].getNeuralLayerType();
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

//gives structure containing learning hyperparameter values
hyperParameters NeuralNetwork::getLearningParameters()
{
	return learningParameters;
}

//destructor for NeuralNetworks to ensure complete memory deallocation
NeuralNetwork::~NeuralNetwork()
{
	delete[] neuralLayers;//causes post-unload-post-intro-exit crash
	delete[] layerStates;

	trainingSamples.clear();
	trainingLabels.clear();
	testingSamples.clear();
	testingLabels.clear();
}

//saves the neural network's state to an xml
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
		layerPropertySubTree.put("activationType", layerStates[i].activationType);
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

//loads a entire neural network's state from an xml
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
	for (const boost::property_tree::ptree::value_type& layer : networkPropertyTree.get_child("network.layers"))
	{
		//defines non-neuron layer state details
		layerStates[i].activationType = layer.second.get<int>("activationType");
		layerStates[i].neuronCount = layer.second.get<int>("neuronCount");

		//defines neuron state details
		for (const boost::property_tree::ptree::value_type& neuron : layer.second.get_child("neurons"))
		{
			//define neuron's saved bias parameter
			layerStates[i].biasOfNeurons.push_back(neuron.second.get<double>("bias"));

			//define neuron's saved weight parameters, skipping the first layer's weights to avoid get_child exception
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
	NeuralNetwork* loadedNetwork = new NeuralNetwork(networkDepth, inputLength, 1, outputLength, errorFunction, layerStates, learningParameters);

	delete[] layerStates;

	return loadedNetwork;
}

//returns the index of the most positive vector element
int getIndexOfMaxEntry(std::vector<double> Vector)
{
	double maxValue = -DBL_MAX;
	int maxIndex = -1;

	for (auto i = 0; i < Vector.size(); i++)
	{
		if (Vector[i] > maxValue)
		{
			maxIndex = i;
			maxValue = Vector[i];
		}
	}

	if (maxIndex < 0)
		throw std::out_of_range("All activations resulted in overflow/underflow");

	return maxIndex;
}

//returns the value of the most positive vector element, todo: temp function
int getValueOfMaxEntry(std::vector<double> Vector)
{
	double maxValue = -DBL_MAX;
	int maxIndex = -1;

	for (auto i = 0; i < Vector.size(); i++)
	{
		if (Vector[i] > maxValue)
		{
			maxIndex = i;
			maxValue = Vector[i];
		}
	}

	if (maxIndex < 0)
		throw std::out_of_range("All activations resulted in overflow/underflow");

	return maxValue;
}

//returns the value of the most negative vector element, todo: temp function
int getValueOfMinEntry(std::vector<double> Vector)
{
	double minValue = DBL_MAX;
	int minIndex = -1;

	for (auto i = 0; i < Vector.size(); i++)
	{
		if (Vector[i] < minValue)
		{
			minIndex = i;
			minValue = Vector[i];
		}
	}

	if (minIndex < 0)
		throw std::out_of_range("All activations resulted in overflow/underflow");

	return minValue;
}

//Final print before leaving menus
void exitSelection()
{
	std::cout << std::endl;
	std::cout << "Exiting manager..." << std::endl;
}

//lists main menu options and prompts user to select one
MenuStates mainSelection(NeuralNetwork*& network)
{
	int selection;

	//ensures old networks being pointed to from previous management menu are deallocated
	delete network;
	network = nullptr;

	//print main menu
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

//prints description of project and provides a high-level user guide regarding main menu options
MenuStates introSelection()
{
	int selection;

	//prints project introduction
	std::cout << std::endl;
	std::cout << "Introduction:" << std::endl;
	std::cout << "Welcome to NeuralNetArchitect! In this console application you can create your own linear neural";
	std::cout << "network with full model structure and optimization algorithm customizability. Currently, only the ";
	std::cout << "MSE cost function and linear neuron activation functions are available. Datasets can manually input into ";
	std::cout << "the network and learning can only be achieved through the editing of the main method. This menu is a ";
	std::cout << "work in progress:)" << std::endl;

	//prints description of main menu options
	std::cout << std::endl;
	std::cout << "Create Neural Network: Create new neural network from scratch" << std::endl;
	std::cout << "Load Neural Network: Load previously saved neural network from project directory" << std::endl;
	std::cout << "Introduction and Info: Where we are at now, introduction and option descriptions" << std::endl;
	std::cout << "Exit Network Manager: Terminate program execution cleanly" << std::endl;

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
	learningParameters.learningRate = 0.00001;
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

	//initialize input layer
	layerDetails[0].activationType = 1;
	layerDetails[0].neuronCount = inputLength * inputWidth;

	//define each layer
	for (int i = 1; i < numberOfLayers; i++)
	{
		std::cout << std::endl << "Define neural layer " << i + 1 << ":\n";

		//define activation type
		std::cout << "\tActivation type: ";
		std::cin >> layerDetails[i].activationType;
		std::cout << std::endl;

		//defines hidden layers
		if (i + 1 < numberOfLayers)
		{
			std::cout << "\tNeuron count: ";
			std::cin >> layerDetails[i].neuronCount;
			std::cout << std::endl;
		}
		//defines output layer
		else
		{
			layerDetails[i].neuronCount = outputCount;
		}

	}

	//create network and point to intialized NeuralNetwork
	*network = new NeuralNetwork(numberOfLayers, inputLength, inputWidth, outputCount, costSelection, layerDetails, learningParameters);

	delete[] layerDetails;

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

//lists manager menu options and prompts user to select one
MenuStates manageSelection()
{
	int selection;

	//display the manageSelection options
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

//asks user for dataset label and sample files and loads them into vectors
MenuStates datasetSelection(NeuralNetwork* network)
{
	std::string trainingImageFilePath, trainingLabelFilePath, testingImageFilePath, testingLabelFilePath;

	//prompts user for dataset directories
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

	network->updateNormalizers();

	//returns to manage menu
	return MenuStates::Manage;
}

//asks user to define higher-level hyperparameters and commences training
MenuStates trainingSelection(NeuralNetwork* network)
{
	int selection;

	try
	{
		network->train();
	}
	catch (DatasetNotLoadedException exception)
	{
		std::cout << std::endl << "Caught DatasetNotLoadedException" << std::endl;
		std::cout << exception.what();

		std::cout << std::endl << "Type 0 to exit:" << std::endl;
		std::cin >> selection;

		return MenuStates::Manage;
	}
	catch (DatasetMismatchException exception)
	{
		std::cout << std::endl << "Caught DatasetMismatchException" << std::endl;
		std::cout << exception.what();

		std::cout << std::endl << "Type 0 to exit:" << std::endl;
		std::cin >> selection;

		return MenuStates::Manage;
	}

	std::cout << std::endl << "Type 0 to exit:" << std::endl;
	std::cin >> selection;

	return MenuStates::Manage;
}

//completes testing of neural network with latest learned-parameter values
MenuStates testingSelection(NeuralNetwork* network)
{
	int selection;
	try
	{
		network->test();
	}
	catch (DatasetNotLoadedException exception)
	{
		std::cout << std::endl << "Caught DatasetNotLoadedException" << std::endl;
		std::cout << exception.what();

		std::cout << std::endl << "Type 0 to exit:" << std::endl;
		std::cin >> selection;

		return MenuStates::Manage;
	}
	catch (DatasetMismatchException exception)
	{
		std::cout << std::endl << "Caught DatasetMismatchException" << std::endl;
		std::cout << exception.what();

		std::cout << std::endl << "Type 0 to exit:" << std::endl;
		std::cin >> selection;

		return MenuStates::Manage;
	}

	std::cout << std::endl << "Type 0 to exit:" << std::endl;
	std::cin >> selection;

	return MenuStates::Manage;
}

//asks user for path of file to store fully-defined neural network in its current state
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

	//prints description of manager menu options
	std::cout << std::endl;
	std::cout << "Select DataSets: Provide directory to load training and testing dataset files" << std::endl;
	std::cout << "Run Training: Begin training the neural network on the training samples" << std::endl;
	std::cout << "Run Testing: Begin testing the neural network on the training samples" << std::endl;
	std::cout << "Save Solution: Save neural network with current parameters as an xml file" << std::endl;
	std::cout << "Help: Where we are at now. Descriptions for the manager menu options" << std::endl;
	std::cout << "Back: Unload neural network from memory and return to creation menu" << std::endl;

	std::cout << std::endl << "Type any integer to exit: ";
	std::cin >> selection;
	return MenuStates::Manage;
}

//indicates error if an invalid menu state is somehow reached
MenuStates defaultSelection()
{
	std::cout << std::endl;
	std::cout << "If we got here, it's a bug... Returning to Main Menu." << std::endl;
	return MenuStates::Main;
}

//contains full fuctionality of neural network manager finite state menu
void manageNeuralNetwork()
{
	NeuralNetwork* network = nullptr;
	MenuStates menuFSMState = MenuStates::Main;

	while (menuFSMState != MenuStates::Exit)
	{
		switch (menuFSMState)
		{

			//Exits menu FSM
		case MenuStates::Exit:
			exitSelection();
			return;

			//Enters main menu
		case MenuStates::Main:
			menuFSMState = mainSelection(network);
			break;

			//Enters introduction page
		case MenuStates::Intro:
			menuFSMState = introSelection();
			break;

			//Enters NeuralNetwork creation sequence
		case MenuStates::Create:
			menuFSMState = createSelection(&network);
			break;

			//Ask for file and load NeuralNetwork
		case MenuStates::Load:
			menuFSMState = loadSelection(&network);
			break;

			//Enters manage menu
		case MenuStates::Manage:
			menuFSMState = manageSelection();
			break;

			//Enters dataset selection
		case MenuStates::Dataset:
			menuFSMState = datasetSelection(network);
			break;

			//Begins training by using training dataset
		case MenuStates::Training:
			menuFSMState = trainingSelection(network);
			break;

			//Begins testing by using testing dataset
		case MenuStates::Testing:
			menuFSMState = testingSelection(network);
			break;

			//Saves NeuralNetwork to xml
		case MenuStates::Save:
			menuFSMState = saveSelection(network);
			break;

			//Enters help page
		case MenuStates::Help:
			menuFSMState = helpSelection();
			break;

			//Print an error if this state is reached
		default:
			menuFSMState = defaultSelection();
			break;
		}
	}
}

InvalidInputException::InvalidInputException(const char* message)
{
	this->message = message;
}

const char* InvalidInputException::what()
{
	return message;
}

InvalidSelectionException::InvalidSelectionException(const char* message)
{
	this->message = message;
}

const char* InvalidSelectionException::what()
{
	return message;
}

DatasetNotLoadedException::DatasetNotLoadedException(const char* message)
{
	this->message = message;
}

const char* DatasetNotLoadedException::what()
{
	return message;
}

DatasetMismatchException::DatasetMismatchException(const char* message)
{
	this->message = message;
}

const char* DatasetMismatchException::what()
{
	return message;
}
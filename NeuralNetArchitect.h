#ifndef NEURALNETARCHITECT_H
#define NEURALNETARCHITECT_H

/**********************************************************************************************************************************************
 Linear neuron's activation is the sumOfproducts(weights, inputActivations) + bias, or the given input if it is in the input layer

  neuronInputListCount; number of neurons with activation that feeds into this neuron's activation function
  inputNeurons; array of addresses of input neurons with an activation that forms part of this neuron's activation
  activation; the evaluation of this neuron's activation function = sumOfproducts(weights, inputActivations) + bias, or the input value
  activationNudgeSum; measurement of how this activation affects cost function, found by sum (dC/da)*(da/da_this) from proceeding neurons
  weights; array of learned weights that are used to modify impact of input neuron activations on this neuron's activation
  bias; the learned negative of the activation threshold that the sumOfProducts needs to surpass to have a positive activation
 **********************************************************************************************************************************************/
class Neuron
{

private:
	int neuronInputListCount;
	Neuron* inputNeurons;
	double activation, activationNudgeSum;
	double* weights, * weightsMomentum;
	double bias, biasMomentum;

protected:
	//gives neuron's internal sumproduct
	double getActivationFunctionInput() const;

	//gives how much this neuron's activation affects the evaluation of the network's cost function
	double getActivationNudgeSum() const;

	//gives how much a particular input affects the evaluation of the network's cost function
	virtual double getActivationRespectiveDerivation(const int inputNeuronIndex) const;

	//gives how much a particular weight affects the evaluation of the network's cost function
	virtual double getWeightRespectiveDerivation(const int inputNeuronIndex) const;

	//gives how much the neuron's bias affects the evaluation of the network's cost function
	virtual double getBiasRespectiveDerivation() const;

	//Considers desired change to activation that would minimize network's cost or error
	void nudgeActivation(double nudge);

public:
	//constructor called for input neurons of activations determined only by direct input
	Neuron();

	//constructor called for hidden neurons during network creation
	Neuron(int neuronInputListCount, Neuron* inputNeurons);

	//constructor called for hidden neurons during network loading, with previously-stored parameter values passed in
	Neuron(int neuronInputListCount, Neuron* inputNeurons, std::vector<double> weightValues, double biasValue);

	//copy constructor for neurons for copying an existing neuron's state
	Neuron(const Neuron& original);

	//operator = overloading for copying an existing neuron's state
	Neuron& operator=(const Neuron& original);

	//custom destructor for neurons to delete neuron
	~Neuron();

	//Defines (lack of) exterior activation function of linear neuron
	virtual void activate(const double input = 0.0);

	//Injects error indicating how much the network's cost function changes according to this neuron's activation
	void setError(double cost);

	//Injects corresponding error into input neurons depending on how their activation affects this neuron's activation
	void injectInputRespectiveCostDerivation() const;

	//Applies change to weights
	void updateWeights(int batchSize, double learningRate, double momentumRetention);

	//Applies change to bias
	void updateBias(int batchSize, double learningRate, double momentumRetention);

	//Resets indication of how this neuron's activation affects the network's cost function evaluation
	void resetNudges();

	//gives number of input neurons
	int getInputCount() const;

	//gives activation value of neuron
	double getActivation() const;

	//gives value of specified weight parameter belonging to this neuron
	double getWeight(int inputNeuronIndex) const;

	std::vector<double> getWeights() const;

	//gives value of this neuron's bias parameter
	double getBias() const;

	//gives the activation type of the neuron
	virtual std::string getNeuronType();

};

/**********************************************************************************************************************************************
 NeuralLayer's activation is either the vector of the outputs of its neurons, or the input values that were directly passed in

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
	void setError(double costArray[]);

	//nudge input layer activations with appropriate derivatives of cost function dC/da * da/di
	void injectErrorBackwards();

	//apply learned weights and bias updates
	void updateParameters(int batchSize, double learningRate, double momentumRetention);

	//clears all stored nudges to neuron parameters
	void clearNudges();

public:
	//default constructor for layer class
	NeuralLayer();

	//constructor called for input layers
	NeuralLayer(int inputLength, int inputWidth);

	//constructor called for hidden layers during network creation, with optional momentum parameter
	NeuralLayer(int neuronCount, NeuralLayer* inputLayer);

	//constructor called for hidden layers during network loading, with stored weights and bias values passed in
	NeuralLayer(int neuronCount, NeuralLayer* inputLayer, std::vector<std::vector<double>> weightValues, std::vector<double> biasValues);

	//copy constructor for layers
	NeuralLayer(const NeuralLayer& original);

	//operator = overloading for readable assignments resulting in deep copies
	NeuralLayer& operator=(const NeuralLayer& original);

	//custom destructor for NeuralLayer objects
	~NeuralLayer();

	//activate all neurons in layer and resets nudges from past learning iteration
	void propagateForward(double inputValues[] = nullptr);

	//transmit error to input neurons and apply learned parameter updates
	void propagateBackward(int batchSize, double learningRate, double momentumRetention, double* costArray = nullptr);

	//returns number of neurons contained within a column of the layer
	int getNeuronArrayLength() const;

	//returns number of neurons contained within a row of the layer
	int getNeuronArrayWidth() const;

	//returns number of neurons contained within layer
	int getNeuronArrayCount() const;

	//returns array of pointers to neurons contained within layer
	Neuron* getNeurons() const;

	//returns pointer to layer that is feeding into this layer
	NeuralLayer* getPreviousLayer() const;

	std::vector<double> getNeuronActivations() const;

	std::vector<std::vector<double>> getNeuronWeights() const;

	std::vector<double> getNeuronBiases() const;

	//returns the activation type of the neurons contained within layer
	virtual int getNeuralLayerType() const;

};

//the derivation of the mean-squared-error function in respect to the activation of an output neuron
double derivedMSECost(double targetValue, double estimatedValue, int outputCount);

//structure used for passing layer details during network creation
struct layerCreationInfo
{
	int type;
	int neuronCount;
};

//structure used for passing layer details during network loading
struct layerLoadingInfo
{
	int type;
	int neuronCount;
	std::vector<std::vector<double>> weightsOfNeurons;
	std::vector<double> biasOfNeurons;
};

struct hyperParameters
{
	double learningRate;
	double learningDecay;
	double batchSize;
	double epochCount;
	double momentumRetention;
	double dropoutPercent;
	double outlierMinError;
	double earlyStoppingMaxError;
};

//flips byte ordering of input integer
unsigned int flipIntegerByteOrdering(int original);

//todo: pass in a path string instead of a bool
//returns vector of all available testing or training labels in the dataset
std::vector<unsigned char> getMNISTLabelVector(bool testing);

//todo: pass in a path string instead of a bool
//returns vector of all available testing or training samples in the dataset
std::vector<std::vector<std::vector<unsigned char>>> getMNISTImageVector(bool testing);


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
	double (*derivedCostFunction)(double, double, int);
	layerLoadingInfo* layerStates;
	hyperParameters learningParameters;
	std::vector<std::vector<std::vector<unsigned char>>> trainingSamples;
	std::vector<unsigned char> trainingLabels;
	std::vector<std::vector<std::vector<unsigned char>>> testingSamples;
	std::vector<unsigned char> testingLabels;


public:

	//constructor for creating NeuralNetworks
	NeuralNetwork(int layerCount, int inputLength, int inputWidth, int outputCount, int costSelection, layerCreationInfo* layerDetails, hyperParameters learningParameters);

	//constructor for loading NeuralNetworks
	NeuralNetwork(int layerCount, int inputLength, int inputWidth, int outputCount, int costSelection, layerLoadingInfo* layerDetails, hyperParameters learningParameters);

	//returns a vector of the activation values of the final layer of the network
	std::vector<double> getOutputs();

	//activates all layers in order from input to output layers
	void propagateForwards(double* inputMatrix);

	//updates parameters in all layers in order from output to input layers
	void propagateBackwards(double* costArray);

	//changes number of samples network expects to process before being told to learn
	void updateBatchSize(int newBatchSize);

	//updates magnitude of parameter changes during learning
	void updateLearningRate(int newLearningRate);

	//loads training samples from dataset
	void updateTrainingSamples();

	//loads training labels from dataset
	void updateTrainingLabels();

	//loads testing samples from dataset
	void updateTestingSamples();

	//loads testing labels from dataset
	void updateTestingLabels();

	//indicates if dataset training samples and labels have been loaded
	bool isReadyForTraining();

	//indicates if dataset testing samples and labels have been loaded
	bool isReadyForTesting();

	//returns dataset training samples to use during network training
	std::vector<std::vector<std::vector<unsigned char>>> getTrainingSamples();

	//returns dataset training labels to use during network training
	std::vector<unsigned char> getTrainingLabels();

	//returns dataset testing samples to use during network testing
	std::vector<std::vector<std::vector<unsigned char>>> getTestingSamples();

	//returns dataset testing labels to use during network testing
	std::vector<unsigned char> getTestingLabels();

	//gives the partial derivative value of the cost function in respect to an output activation
	double getOutputRespectiveCost(double targetValue, int outputIndex);

	//gives the number of inputs that network accepts
	int getInputCount();

	//gives the number of outputs that the network produces
	int getOutputCount();

	//gives the depth of the network
	int getLayerCount();

	//saves all necessary layer data necessary to recreate the network layers exactly upon loading
	void saveLayerStates();

	//gives array of layer state details that may be used to recreate layers
	layerLoadingInfo* getLayerStates();

	hyperParameters getLearningParameters();

	~NeuralNetwork();

};

//saves the entire neural network to an xml, such that all data necessary to rebuild the exact network is stored
void storeNetwork(NeuralNetwork* network, std::string& fileName);

//saves the entire neural network to an xml, such that all data necessary to rebuild the exact network is stored
NeuralNetwork* loadNetworkPointer(const std::string& fileName);

//returns the index of the most positive vector element
int getIndexOfMaxEntry(std::vector<double> Vector);

//returns the value of the most positive vector element
int getValueOfMaxEntry(std::vector<double> Vector);

//returns the value of the most negative vector element
int getValueOfMinEntry(std::vector<double> Vector);

//enumeration to number-code the names of the menu states
enum class MenuStates : unsigned int
{
	Exit = 0,
	Main = 1,
	Intro = 2,
	Create = 3,
	Load = 4,
	Manage = 5,
	Dataset = 6,
	Training = 7,
	Testing = 8,
	Save = 9,
	Help = 10,
};

//Final print before leaving menus
void exitSelection();

//lists main menu options and prompts user to select one
MenuStates mainSelection(NeuralNetwork* network);

//todo: improve this
//prints description of project and provides a high-level guide
MenuStates introSelection();

//prompts user through creation of a neural network
MenuStates createSelection(NeuralNetwork** network);

//asks user for path of file to load fully-defined neural network from
MenuStates loadSelection(NeuralNetwork** network);

//lists manager options and prompts user to select one
MenuStates manageSelection();

//asks user for datatset label and sample files and loads them into vectors
MenuStates datasetSelection(NeuralNetwork* network);

//asks user to define higher-level hyperparameters and commences training
MenuStates trainingSelection(NeuralNetwork* network);

//completes testing of neural network with current learned-parameter values
MenuStates testingSelection(NeuralNetwork* network);

//asks user for path of file to store fully-defined neural network in
MenuStates saveSelection(NeuralNetwork* network);

//prints detailed instructions and explanation on the customization options
MenuStates helpSelection();

//indicates error if an invalid menu state is somehow reached
MenuStates defaultSelection();

//contains full fuctionality of neural network manager Finite State Menu
void manageNeuralNetwork();

#endif
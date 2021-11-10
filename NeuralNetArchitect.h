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

	//custom destructor for deleting Neuron objects
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

	//Set error of neurons with activations directly used to calculate cost
	void setError(double costArray[]);

	//nudge preceeding layer activation depending on how much they affect the network's cost function evaluation
	void injectErrorBackwards();

	//apply learned weights and bias updates
	void updateParameters(int batchSize, double learningRate, double momentumRetention);

	//clears all stored nudges to layer's activation
	void clearNudges();

public:
	//default constructor for layer class, todo: get rid of this?
	NeuralLayer();

	//constructor called for creating input layers
	NeuralLayer(int inputLength, int inputWidth);

	//constructor called for hidden layers during network creation
	NeuralLayer(int neuronCount, NeuralLayer* inputLayer);

	//constructor called for hidden layers during network loading, with stored neuron parameter values passed in
	NeuralLayer(int neuronCount, NeuralLayer* inputLayer, std::vector<std::vector<double>> weightValues, std::vector<double> biasValues);

	//copy constructor for creating layers of the same state as the one passed in
	NeuralLayer(const NeuralLayer& original);

	//operator = overloading for creating layers of the same state as the one passed in
	NeuralLayer& operator=(const NeuralLayer& original);

	//custom destructor for deleting NeuralLayer objects
	~NeuralLayer();

	//activate layer based on previous layer's activation
	void propagateForward(double inputValues[] = nullptr);

	//inject error vector to previous layer depending on this layer's error vector
	void propagateBackward(int batchSize, double learningRate, double momentumRetention, double* costArray = nullptr);

	//gives number of neurons contained within a column of this layer
	int getNeuronArrayLength() const;

	//gives number of neurons contained within a row of this layer
	int getNeuronArrayWidth() const;

	//gives number of neurons contained within layer
	int getNeuronArrayCount() const;

	//gives array of addresses to neurons contained within layer
	Neuron* getNeurons() const;

	//gives address of layer that is feeding into this layer
	NeuralLayer* getPreviousLayer() const;

	//get the activations of this layer's neurons
	std::vector<double> getNeuronActivations() const;

	//get the weight values stored in each neuron in this layer
	std::vector<std::vector<double>> getNeuronWeights() const;

	//get the bias value corresponding to each neuron in this layer
	std::vector<double> getNeuronBiases() const;

	//gives this layer's activation function
	virtual int getNeuralLayerType() const;

};

//todo: change this to bool determining derived state
double derivedMSECost(double targetValue, double estimatedValue, int outputCount);

/**********************************************************************************************************************************************
 layerCreationInfo is used for passing layer details for NeuralNetwork creation

  type; activation type of layer, the activation function that determines each neuron's output
  neuronCount; number of neurons stored in layer
 **********************************************************************************************************************************************/
struct layerCreationInfo
{
	int type;
	int neuronCount;
};

/**********************************************************************************************************************************************
 layerLoadingInfo is used for passing layer details for NeuralNetwork loading

  type; activation type of layer, the activation function that determines each neuron's output
  neuronCount; number of neurons stored in layer
  weightsOfNeurons; loaded weight parameter values of neurons
  biasOfNeurons; loaded bias parameter values of neurons
 **********************************************************************************************************************************************/
struct layerLoadingInfo
{
	int type;
	int neuronCount;
	std::vector<std::vector<double>> weightsOfNeurons;
	std::vector<double> biasOfNeurons;
};

/**********************************************************************************************************************************************
 hyperParameters is used for storing values of NeuralNetwork's learning hyperparameters

  learningRate; the portion of a learning step that is taken when updating parameters during backpropagation
  learningDecay; the rate at which learning step sizes cumulatively decrease after each batch is learned on
  batchSize; the number of samples used to determine the magnitude and averaged direction of a learning step
  epochCount; the number of times that the entire training dataset is used for learning
  momentumRetention; the amount that a learning step influences the magnitude and direction of the next step
  dropoutPercent; the rate of neurons in the network that will not be activated or trained during an epoch
  outlierMinError; the criteria by which an example is deemed an outlier of the dataset and will not be trained on
  earlyStoppingMaxError; the criteria used to determined whether a network has reached its learning goals after an epoch
 **********************************************************************************************************************************************/
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
//gives all available testing or training labels in the dataset
std::vector<unsigned char> getMNISTLabelVector(bool testing);

//todo: pass in a path string instead of a bool
//gives all available testing or training samples in the dataset
std::vector<std::vector<std::vector<unsigned char>>> getMNISTImageVector(bool testing);


/**********************************************************************************************************************************************
 NeuralNetworks's activation is a function of all weights and bias parameters held within the neurons of each layer

  layerCount; number of neural layers held within neural network, the depth of the network
  inputLength; the first dimension defining the size of the input array, currently assuming a 2D input grid
  inputWidth; the second dimension defining the size of the input array, currently assuming a 2D input grid
  outputCount; the number of outputs the neural network is expected to produce, currently assuming a vector output
  neuralLayers; an array containing all neural layers that make up the network
  derivedCostFunction; address of function... todo: update this after fixing function with bool
  layerStates; saved states of layers, containing everything that is needed to fully define and reconstruct the layers
  learningParameters; learning hyperparameters that tweak how the network will carry out training
  trainingSamples; vector of raw samples from the dataset that will be used to train the network
  trainingLabels; vector of labels that correctly classify the training samples used to train the network
  testingSamples; vector of raw samples from the dataset that will be used to test the network
  testingLabels; vector of labels that correctly classify the training samples used to test the network
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

	//gives a vector of the activation values of the network's final layer
	std::vector<double> getOutputs();

	//activates all layers in order from input to output layers
	void propagateForwards(double* inputMatrix);

	//performs learning step in all layers in order from output to input layers
	void propagateBackwards(double* costArray);

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

	//gives dataset training samples to use during network training
	std::vector<std::vector<std::vector<unsigned char>>> getTrainingSamples();

	//gives dataset training labels to use during network training
	std::vector<unsigned char> getTrainingLabels();

	//gives dataset testing samples to use during network testing
	std::vector<std::vector<std::vector<unsigned char>>> getTestingSamples();

	//gives dataset testing labels to use during network testing
	std::vector<unsigned char> getTestingLabels();

	//gives the partial derivative value of the cost function in respect to an output activation
	double getOutputRespectiveCost(double targetValue, int outputIndex);

	//gives the number of inputs that the network accepts
	int getInputCount();

	//gives the number of outputs that the network produces
	int getOutputCount();

	//gives the depth of the network
	int getLayerCount();

	//saves all necessary layer data necessary to accurately recreate the network layers upon loading
	void saveLayerStates();

	//gives array of layer state details that may be used to recreate layers
	layerLoadingInfo* getLayerStates();

	hyperParameters getLearningParameters();

	~NeuralNetwork();

};

//saves the entire neural network, such that all data necessary to rebuild the exact network is stored
void storeNetwork(NeuralNetwork* network, std::string& fileName);

//loads the entire neural network, such that all data necessary to rebuild the exact network is restored
NeuralNetwork* loadNetworkPointer(const std::string& fileName);

//gives the index of the most positive vector element
int getIndexOfMaxEntry(std::vector<double> Vector);

//gives the value of the most positive vector element
int getValueOfMaxEntry(std::vector<double> Vector);

//gives the value of the most negative vector element
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

//prints performed when leaving menus
void exitSelection();

//lists main menu options and prompts user to select one
MenuStates mainSelection(NeuralNetwork* network);

//todo: improve this
//prints description of project and provides a high-level guide
MenuStates introSelection();

//prompts user through creation of a neural network
MenuStates createSelection(NeuralNetwork** network);

//asks user for path of file to load fully-defined neural network
MenuStates loadSelection(NeuralNetwork** network);

//lists manager options and prompts user to select one
MenuStates manageSelection();

//asks user for datatset label and sample files and loads them
MenuStates datasetSelection(NeuralNetwork* network);

//commences training
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
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

//main execution of program
int main()
{
	manageNeuralNetwork();

	//todo: Input Neuron to Logistic Neuron, and ReLU neuron unit tests
	//Neuron neuron = LogisticNeuron();

	return 0;
}
// 784 10 2 1 0.0001 0 1 1 0 0 1 0 1			MNIST linear network			(75% success asymptote)
// 784 10 2 1 0.0001 0 1 1 0 0 1 0 2			MNIST ReLU network				(20% success asymptote)
// 784 10 2 2 0.0001 0 1 1 0 0 1 0 3			MNIST In->10Sig with BCE		(45% success asymptote, 10^-5)
// 784 10 3 2 0.0001 0 1 1 0 0 1 0 2 20 3		MNIST In->20ReLU->10Sig			(10% success declining)
// 784 10 3 2 0.0001 0 1 1 0 0 1 0 5 10 6		MNIST In->Softmax with BCE		(2% success)

/* Cleanup todo:
* Move to .hpp and .cpp file setup																			--DONE
* Make all hyperparameters (learning rate, batch size, momentum...) be stored in NeuralNetwork				--DONE
* Remove all hyperparameters from sub-network classes, pass in hyperparameters from network by arguments	--DONE
* Create NeuralNetwork Destructor methods and write calls for memory management								--DONE
* ----------------------------------------------------------------------------------------------------------
* Write 'how' comment above each definition, 'what' comment on declarations, comments summarizing blocks	
* Delete unused functions
* Change names of confusing functions
* Scan for memory leaks
* Identify potential space and time complexity reductions
* ----------------------------------------------------------------------------------------------------------
* Accomodate for invalid input handling
* Polish user input prompts and comments
* Find sections where exception detection handling should be carried out
* Finish Help and Introduction prompts
* ----------------------------------------------------------------------------------------------------------
* Update project description on Github and repo permissions
* Decide and enforce consistent formatting of all code
* Enable all commented out, red-lined, or hard-coded features
* Expand menu functions to acknowledge newly-enabled features
* ----------------------------------------------------------------------------------------------------------
* Require check status before being able to merge to Github repo
* Use submodule of all 3rd party libraries
* Ensure ease of use between Visual Studio project and Github repo
* Test that entire project can be identically used on another system
* ----------------------------------------------------------------------------------------------------------
* Fix randomization of learned parameters during network creation... training results are sometimes the same, even in repetition
*/

/* Expansion todo:
* Logistic neuron/layer
* ReLU neuron/layer
* ...
* Softplus neuron/layer
* Step neuron/layer
* Recurrent neuron/layer
* Convolutional neuron/layer
* Softmax neuron/layer
* Input (pre) processing ...
* Output (post) processing ...
* Dropout learning feature
* Early-stopping learning feature
* All hyperparameter effects
* Edge-case detection learning feature
* Parallel programming
* Support Vector Machines
* K Nearest Neighbors
* Decision Stumps
* Boosting (Ensembling)
*/

/* NC notes: 
* Fix deterministic training for linear models, improve accuracy of training
* Provide debugging entry to neuralNetwork saving - activation value
* Begin forensic saving at some point
* Neuron class inheritance for sigmoid and ReLU -- DONE
* LogisticNeuron and ReLUNeuron unit tests, layer test, network tests
* Separate functions into multiple .h and .cpp files
* Dataset getters used as least as possible -> training NeuralNetwork internal function
* NeuralLayerDetails.type in switch statement in layer creation/loading for feedforward Neurons
* Exploding output problem when training on loaded NN
*/

/* Derived Neurons todo list:
* Define all necessary LogisticNeuron and ReLUNeuron constructors (Neurons)
* Relocate NeuralNetwork layer type switch statement to NeuralLayer
* Activation function enum (linear, sig, ReLU) -> !!!
* Ensure nothing is being left out from derived classes
* LogisticNeuron and ReLUNeuron unit tests -> !!!
* Creation menu flexibility for derived Neuron classes
* Ensure loading menu functionality for derived Neuron Classes
* Test LinearInput->ReLU||Logistic NeuralNetwork on MNIST -> !!!
* Ensure proper Neuron* to std::vector<Neuron*> conversion -> !!!
* Fix Overflow/Underflow issues preventing learning
*/
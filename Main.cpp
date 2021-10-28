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

	return 0;
}
// 2 1 4 1 1 0 1 2 0 1 0	the first network		(propagation checking)
// 2 2 4 1 1 0 1 2 0 1 0	the usual network		(loading + learning correctness)
// 784 10 2 0 0				MNIST linear network	(dataset confirmation)

/* Cleanup todo:
* Move to .hpp and .cpp file setup
* Make all hyperparameters (learning rate, batch size, momentum...) be stored only in NeuralNetwork
* Create Destructor methods and write calls for memory management
* Write 'how' comment above each definition, 'what' comment on declarations, comments summarizing blocks of code
* Decide and enforce consistent formatting of all code
* Enable all commented out, red-lined, or hard-coded features
* Expand menu functions to acknowledge newly-enabled features
* Accomodate for invalid input handling
* Find sections where exception detection handling should be carried out
* Identify potential space and time complexity reductions
* Delete unused functions
* Ensure ease of use between Visual Studio project and Github repo
* Update project description on Github
* Change names of confusing functions after looking up 'sweet spot' of description depth
*/

/* Expansion todo:
* Sigmoid neuron/layer
* ReLU neuron/layer
* ...
* Softplus neuron/layer
* Step neuron/layer
* Recurrent neuron/layer
* Convolutional neuron/layer
* Softmax neuron/layer
* Input processing ...
* Output processing ...
* Dropout learning feature
* Early-stopping learning feature
* Edge-case detection learning feature
*/
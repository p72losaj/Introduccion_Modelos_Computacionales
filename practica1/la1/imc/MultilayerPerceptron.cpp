/*********************************************************************
* File  : MultilayerPerceptron.cpp
* Date  : 2020
*********************************************************************/

#include "MultilayerPerceptron.h"

#include "util.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <math.h>


using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Constructor: Default values for all the parameters -> Hecho
MultilayerPerceptron::MultilayerPerceptron()
{
	this->nOfLayers = 1;
	this->eta = 0.1;
	this->mu = 0.9;
}

// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[]) {
	this->nOfLayers = nl;
	this->layers = new Layer[nl];

	for (int i = 0; i < nl; i++) {
		this->layers[i].nOfNeurons = npl[i];
		this->layers[i].neurons = new Neuron[npl[i]];
	}

	for (int i = 0; i < nl; i++) {
		for (int j = 0; j < npl[i]; j++) {
			if (i == 0) {
				this->layers[i].neurons[j].w = new double[npl[i]];
				this->layers[i].neurons[j].deltaW = new double[npl[i]];
				this->layers[i].neurons[j].wCopy = new double[npl[i]];
			}
			else {
				this->layers[i].neurons[j].w = new double[npl[i - 1]];
				this->layers[i].neurons[j].deltaW = new double[npl[i - 1]];
				this->layers[i].neurons[j].wCopy = new double[npl[i - 1]];
			}
		}
	}
	return 1;
}


// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron() {
	freeMemory();
}


// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory() {
	for(int i=0; i < this->nOfLayers; i++){
		for(int j=0; j<layers[i].nOfNeurons; j++){
			if(this->layers[i].neurons[j].w != NULL){
				delete[] this->layers[i].neurons[j].w;
			}
			if(this->layers[i].neurons[j].deltaW != NULL){
				delete[] this->layers[i].neurons[j].deltaW;
			}
			if(this->layers[i].neurons[j].lastDeltaW != NULL){
				delete[] this->layers[i].neurons[j].lastDeltaW;
			}
			if(this->layers[i].neurons[j].wCopy != NULL){
				delete[] this->layers[i].neurons[j].wCopy;
			}
		}
		delete[] this->layers[i].neurons;
	}
	delete[] this->layers;

}

// ------------------------------
// Feel all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights() {
	// TODO: Implementar
	for(int i=0; i < this->nOfLayers; i++){
		for(int j=0; j<layers[i].nOfNeurons; j++){
			if(this->layers[i].neurons[j].w != NULL){
				for(int k=0; k<layers[i-1].nOfNeurons; k++){
					this->layers[i].neurons[j].w[k] = util::randomDouble(-1.0, 1.0);
				}
			}
		}
	}

}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(double* input) {
	for(int i=0; i<this->layers[0].nOfNeurons; i++){
		this->layers[0].neurons[i].out = input[i];
	}
}

// ------------------------------
// Get the outputs predicted by the network (out vector the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(double* output)
{
	for(int i=0; i<this->layers[this->nOfLayers-1].nOfNeurons; i++){
		output[i] = this->layers[this->nOfLayers-1].neurons[i].out;
	}
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights() {
	for(int i=1; i<this->nOfLayers; i++){
		for(int j=0; j<this->layers[i].nOfNeurons; j++){
			if(this->layers[i].neurons[j].w != NULL){
				for(int k=0; k<this->layers[i-1].nOfNeurons + 1; k++){
					this->layers[i].neurons[j].wCopy[k] = this->layers[i].neurons[j].w[k];
				}
			}
		}
	}
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights() {
	for(int i=1; i< this->nOfLayers; i++){
		for(int j=0; j<this->layers[i].nOfNeurons; j++){
			for(int k=0; k<this->layers[i-1].nOfNeurons + 1; k++){
				this->layers[i].neurons[j].w[k] = this->layers[i].neurons[j].wCopy[k];
			}
		}
	}
}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate() {
		double net;
		for(int i=1; i< this->nOfLayers; i++){
			for(int j=0; j<this->layers[i].nOfNeurons; j++){
				net = 0.0;
				for(int k=1; k<this->layers[i-1].nOfNeurons; k++){
					net += this->layers[i].neurons[j].w[k] * this->layers[i-1].neurons[k].out;
				}
				net += this->layers[i].neurons[j].w[0];
				this->layers[i].neurons[j].out = 1.0 / (1+exp(-net));
			}
		}
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
double MultilayerPerceptron::obtainError(double* target) {
	double mse = 0.0;
	for(int i=0; i<this->layers[this->nOfLayers-1].nOfNeurons; i++){
		mse += pow(target[i] - this->layers[this->nOfLayers-1].neurons[i].out, 2);
	}
	return mse / this->layers[this->nOfLayers-1].nOfNeurons;
}


// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
void MultilayerPerceptron::backpropagateError(double* target) {
	double out, aux;
	for(int i=0; i < this->layers[this->nOfLayers-1].nOfNeurons; i++){
		out = this->layers[this->nOfLayers-1].neurons[i].out;
		this->layers[this->nOfLayers-1].neurons[i].delta = -(target[i] - out) * out * (1.0 - out);
	}

	for(int i=this->nOfLayers-2; i >= 1; i++){
		for(int j=0; j< this->layers[i].nOfNeurons; j++){
			out = this->layers[i].neurons[j].out;
			aux = 0.0;
			for(int k=0; k< this->layers[i+1].nOfNeurons; k++){
				aux += this->layers[i+1].neurons[k].w[j+1] * this->layers[i+1].neurons[k].delta;
			}
			this->layers[i].neurons[j].delta = aux * out * (1.0 - out);
		}
	}
}


// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange() {
	for(int i=1; i < this->nOfLayers; i++){
		for(int j=0; j<this->layers[i].nOfNeurons; j++){
			for(int k=1; k < this->layers[i-1].nOfNeurons; k++){
				this->layers[i].neurons[j].deltaW[k] += this->layers[i].neurons[j].delta * this->layers[i-1].neurons[k-1].out;
			}
			this->layers[i].neurons[j].deltaW[0] += this->layers[i].neurons[j].delta;
		}
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment() {
	double newEta;
	for(int i=1; i < this->nOfLayers; i++){
		for(int j=0; j<this->layers[i].nOfNeurons; j++){
			for(int k=1; k < this->layers[i-1].nOfNeurons; k++){
				newEta = this->eta * this->layers[i].neurons[j].deltaW[k];
				this->layers[i].neurons[j].w[k] -= newEta;
				this->layers[i].neurons[j].deltaW[k] = 0.0;
			}
			newEta = this->eta * this->layers[i].neurons[j].deltaW[0];
			this->layers[i].neurons[j].w[0] -= newEta;
			this->layers[i].neurons[j].deltaW[0] = 0.0;
		}
	}

}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork() {
	for(int i=1; i < this->nOfLayers; i++){
		cout << "Layer " << i << endl;
		for(int j=0; j<this->layers[i].nOfNeurons; j++){
			for(int k=0; k < this->layers[i-1].nOfNeurons + 1; k++){
				cout << this->layers[i].neurons[j].w[k] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
void MultilayerPerceptron::performEpochOnline(double* input, double* target) {
	// TODO: implement the online version of the epoch
	cout << "Performing epoch online" << endl;
	// 1. Forward propagate the inputs
	this->forwardPropagate();
	// 2. Backpropagate the error
	this->backpropagateError(target);
	// 3. Accumulate the changes
	this->accumulateChange();
	// 4. Adjust the weights
	this->weightAdjustment();

}

// ------------------------------
// Perform an online training for a specific trainDataset
void MultilayerPerceptron::trainOnline(Dataset* trainDataset) {
	int i;
	for(i=0; i<trainDataset->nOfPatterns; i++){
		performEpochOnline(trainDataset->inputs[i], trainDataset->outputs[i]);
	}
	
}

// ------------------------------
// Test the network with a dataset and return the MSE
double MultilayerPerceptron::test(Dataset* testDataset) {
	double mse = 0.0;
	for(int i=0; i<testDataset->nOfPatterns; i++){
		this->feedInputs(testDataset->inputs[i]);
		this->forwardPropagate();
		mse += this->obtainError(testDataset->outputs[i]);
	}
	return mse/testDataset->nOfPatterns;
}


// Optional - KAGGLE
// Test the network with a dataset and return the MSE
// Your have to use the format from Kaggle: two columns (Id y predictied)
void MultilayerPerceptron::predict(Dataset* pDatosTest)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers-1].nOfNeurons;
	double * obtained = new double[numSalidas];
	
	cout << "Id,Predicted" << endl;
	
	for (i=0; i<pDatosTest->nOfPatterns; i++){

		feedInputs(pDatosTest->inputs[i]);
		forwardPropagate();
		getOutputs(obtained);
		
		cout << i;

		for (j = 0; j < numSalidas; j++)
			cout << "," << obtained[j];
		cout << endl;

	}
}

// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
void MultilayerPerceptron::runOnlineBackPropagation(Dataset * trainDataset, Dataset * pDatosTest, int maxiter, double *errorTrain, double *errorTest)
{
	int countTrain = 0;
	cout << "Run Online Back Propagation" << endl;
	// Random assignment of weights (starting point)

	this->randomWeights(); 
	double minTrainError = 0;
	int iterWithoutImproving;
	double testError = 0;


	// Learning
	do {
		trainOnline(trainDataset);
		double trainError = test(trainDataset);
		if(countTrain==0 || trainError < minTrainError){
			minTrainError = trainError;
			copyWeights();
			iterWithoutImproving = 0;
		}
		else if( (trainError-minTrainError) < 0.00001)
			iterWithoutImproving = 0;
		else
			iterWithoutImproving++;

		if(iterWithoutImproving==50){
			cout << "We exit because the training is not improving!!"<< endl;
			restoreWeights();
			countTrain = maxiter;
		}


		countTrain++;

		cout << "Iteration " << countTrain << "\t Training error: " << trainError << endl;

	} while ( countTrain<maxiter );

	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();

	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for(int i=0; i<pDatosTest->nOfPatterns; i++){
		double* prediction = new double[pDatosTest->nOfOutputs];

		// Feed the inputs and propagate the values
		feedInputs(pDatosTest->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for(int j=0; j<pDatosTest->nOfOutputs; j++)
			cout << pDatosTest->outputs[i][j] << " -- " << prediction[j] << " ";
		cout << endl;
		delete[] prediction;

	}

	testError = test(pDatosTest);
	*errorTest=testError;
	*errorTrain=minTrainError;

}

// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char * archivo)
{
	// Object for writing the file
	ofstream f(archivo);

	if(!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for(int i = 0; i < nOfLayers; i++)
		f << " " << layers[i].nOfNeurons;
	f << endl;

	// Write the weight matrix of every layer
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;

}


// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char * archivo)
{
	// Object for reading a file
	ifstream f(archivo);

	if(!f.is_open())
		return false;

	// Number of layers and number of neurons in every layer
	int nl;
	int *npl;

	// Read number of layers
	f >> nl;

	npl = new int[nl];

	// Read number of neurons in every layer
	for(int i = 0; i < nl; i++)
		f >> npl[i];

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				f >> layers[i].neurons[j].w[k];

	f.close();
	delete[] npl;

	return true;
}

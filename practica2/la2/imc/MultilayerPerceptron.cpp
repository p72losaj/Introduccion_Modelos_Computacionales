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
#include <algorithm> // Para calcular la distancia euclidea
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <limits>
#include <math.h>


using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
	this->nOfLayers = 1;
	this->eta = 0.7;
	this->mu = 1;
	this->online = false; // Por defecto, funcion offline
	this->outputFunction = 0; // Por defecto, funcion sigmoide en la capa de salida
}


// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[]) {
	this->nOfLayers = nl; // Total number of layers in the network
	this->layers = new Layer[nl]; // Vector containing every layer
	
	for(int i=0; i< this->nOfLayers; i++){
		this->layers[i].nOfNeurons = npl[i]; // Number of neurons in the layer
		this->layers[i].neurons = new Neuron[npl[i]]; // Vector with the neurons of the layer
	}
	
	for (int i = 0; i < nl; i++) {
		for (int j = 0; j < npl[i]; j++) {
			if (i == 0) {
				// this->layers[i].neurons[j].w = new double[npl[i]]; // Input weight vector (w_{ji}^h)
				// this->layers[i].neurons[j].deltaW = new double[npl[i]]; // Change to be applied to every weight (\Delta_{ji}^h (t)) 
				// this->layers[i].neurons[j].wCopy = new double[npl[i]]; // Copy of the input weights
				this->layers[i].neurons[j].w = NULL;
				this->layers[i].neurons[j].deltaW = NULL;
				this->layers[i].neurons[j].wCopy = NULL;
				this->layers[i].neurons[j].lastDeltaW = NULL;
			}
			else {
				this->layers[i].neurons[j].w = new double[npl[i - 1] + 1]; // Input weight vector (w_{ji}^h)
				this->layers[i].neurons[j].deltaW = new double[npl[i - 1] + 1]; // Change to be applied to every weight (\Delta_{ji}^h (t))
				this->layers[i].neurons[j].wCopy = new double[npl[i - 1] + 1]; // Copy of the input weights
				this->layers[i].neurons[j].lastDeltaW = new double[npl[i - 1] + 1]; 
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
void MultilayerPerceptron::freeMemory() { // Anadido en practica1
	for(int i=0; i<this->nOfLayers;i++){ // For every layer
		// Free memory for the neurons
		for(int j=0; j<this->layers[i].nOfNeurons; j++){ // For every neuron
			delete[] this->layers[i].neurons[j].w; // Free memory for the input weights
			delete[] this->layers[i].neurons[j].deltaW; // Free memory for the change to be applied to every weight
			delete[] this->layers[i].neurons[j].wCopy; // Free memory for the copy of the input weights
		}
		delete[] this->layers[i].neurons; // Delete the neuron vector

	}
	delete[] this->layers; // Delete the layer vector
}

// ------------------------------
// Fill all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights() { // Anadido en practica1
	for (int i = 1; i < this->nOfLayers; i++) {
		for (int j = 0; j < this->layers[i].nOfNeurons; j++) {
			for (int k = 0; k < this->layers[i-1].nOfNeurons + 1; k++) {
				this->layers[i].neurons[j].w[k] = util::randomDouble(-1, 1);
			}
		}
	}
}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(double* input) { // anadido en practica1
	for (int i = 0; i < this->layers[0].nOfNeurons; i++) {
		this->layers[0].neurons[i].out = input[i];
	}
}

// ------------------------------
// Get the outputs predicted by the network (out vector of the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(double* output)
{ // anadido en practica1
	for (int i = 0; i < this->layers[this->nOfLayers-1].nOfNeurons; i++) {
		output[i] = this->layers[this->nOfLayers-1].neurons[i].out;
	}
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights() { // anadido en practica1
	for (int i = 1; i < this->nOfLayers; i++) {
		for (int j = 0; j < this->layers[i].nOfNeurons; j++) {
			for (int k = 0; k < this->layers[i-1].nOfNeurons + 1; k++) {
				this->layers[i].neurons[j].wCopy[k] = this->layers[i].neurons[j].w[k];
			}
		}
	}
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights() { // Anadido en practica1
	for (int i = 1; i < this->nOfLayers; i++) {
		for (int j = 0; j < this->layers[i].nOfNeurons; j++) {
			for (int k = 0; k < this->layers[i-1].nOfNeurons + 1; k++) {
				this->layers[i].neurons[j].w[k] = this->layers[i].neurons[j].wCopy[k];
			}
		}
	}
}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate() { 
	double net;
	double sumNet = 0.0;

	for(int i=1; i<this->nOfLayers; i++){
		sumNet = 0.0;
		for(int j=0; j<this->layers[i].nOfNeurons; j++){
			net = 0.0;
			for(int k=1; k<this->layers[i-1].nOfNeurons +1; k++){
				net += this->layers[i].neurons[j].w[k] * this->layers[i-1].neurons[k-1].out;
			}

			net += this->layers[i].neurons[j].w[0];

			if((i == (this->nOfLayers - 1)) && (this->outputFunction == 1)){
				this->layers[i].neurons[j].out = exp(net); // Softmax
				sumNet += exp(net);
			}
			else{
				this->layers[i].neurons[j].out = 1.0 / (1 + exp(-net));
			}
		}

		if((i == (this->nOfLayers - 1)) && (this->outputFunction == 1)){
			for(int j=0; j<this->layers[i].nOfNeurons; j++){
				this->layers[i].neurons[j].out /= sumNet;
			}
		}
	}
	
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::obtainError(double* target, int errorFunction) { // Nuevo de la practica -> Incorporar funcion de error L (entropia cruzada)
// Nueva expresion de δ^h_j
	// Error MSE -> Practica1
	
	if(errorFunction == 0){
		double mse = 0.0;
		for(int i=0; i<this->layers[this->nOfLayers-1].nOfNeurons; i++){ 
			mse += pow(target[i] - this->layers[this->nOfLayers - 1].neurons[i].out, 2); // Calculate the error
		}

		return mse /= (double) this->layers[this->nOfLayers - 1].nOfNeurons; // Calculate the mean
	}
	// Error Entropia cruzada
	double ce = 0.0;
	for(int i=0; i<this->layers[this->nOfLayers-1].nOfNeurons; i++){ // For every neuron
		ce += target[i] * log(this->layers[this->nOfLayers - 1].neurons[i].out); // Calculate the error
	}

	return ce / (double) this->layers[this->nOfLayers - 1].nOfNeurons; // Calculate the mean
	
}


// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::backpropagateError(double* target, int errorFunction) { // Nuevo de la practica -> Nueva expresion de δ^h_j )
	double out, aux;

	for(int i=0; i<this->layers[this->nOfLayers-1].nOfNeurons; i++){
		out = this->layers[nOfLayers-1].neurons[i].out;
		this->layers[this->nOfLayers-1].neurons[i].delta = 0.0;
		//Softmax
		if(this->outputFunction == 1){
			int conditionSoftmax = 0;

			for(int j=0; j<this->layers[this->nOfLayers-1].nOfNeurons; j++){
				if(j == i){
					conditionSoftmax = 1;
				}
				else{
					conditionSoftmax = 0;
				}

				if(errorFunction == 0){					
					this->layers[this->nOfLayers - 1].neurons[i].delta += -(target[j] - this->layers[this->nOfLayers-1].neurons[j].out) * out * (conditionSoftmax - this->layers[this->nOfLayers-1].neurons[j].out);
				}
				else{
					this->layers[this->nOfLayers - 1].neurons[i].delta += -(target[j] / this->layers[this->nOfLayers-1].neurons[j].out) * out * (conditionSoftmax - this->layers[this->nOfLayers-1].neurons[j].out);
				}
			}
		}
		//Sigmoid
		else{
			if(errorFunction == 0){
				this->layers[this->nOfLayers - 1].neurons[i].delta = -(target[i] - out) * out * (1 - out);
			}
			else{
				this->layers[this->nOfLayers - 1].neurons[i].delta = -(target[i] / out) * out * (1 - out);
			}
		}
	}

	for(int i=this->nOfLayers-2; i>=1; i--){ // For every layer
		for( int j=0; j<this->layers[i].nOfNeurons; j++){ // For every neuron
			out = this->layers[i].neurons[j].out; // Get the output
			aux = 0.0; // Reset aux
			for(int k=0; k<this->layers[i+1].nOfNeurons; k++){ // For every neuron in the next layer
				aux += this->layers[i+1].neurons[k].w[j+1] * this->layers[i+1].neurons[k].delta; // Calculate the aux
			}

			this->layers[i].neurons[j].delta = aux * out * (1 - out); // Calculate the delta
		}
	}
}
// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange() { // anadido en practica1
	for(int i=1; i<this->nOfLayers; i++){ // Para cada capa

		for(int j=0; j<this->layers[i].nOfNeurons; j++){ // Para cada neurona de la capa i

			for(int k=1; k<this->layers[i-1].nOfNeurons +1; k++){ // Para cada neurona de la capa i-1
				// Acumular el cambio
				this->layers[i].neurons[j].deltaW[k] += this->layers[i].neurons[j].delta * this->layers[i-1].neurons[k-1].out;
			}

			this->layers[i].neurons[j].deltaW[0] += this->layers[i].neurons[j].delta; // Sesgo
		}
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment() { // anadido en practica1
	// modo online
	if(this->online == true){
		for(int i=1; i<this->nOfLayers; i++){ // For every layer
			for(int j=1; j<this->layers[i].nOfNeurons; j++){ // For every neuron
				for(int k=1; k<this->layers[i-1].nOfNeurons +1; k++){ // For every weight
					this->layers[i].neurons[j].w[k] -= (this->eta*this->layers[i].neurons[j].deltaW[k])
						- (this->mu * this->eta * this->layers[i].neurons[j].lastDeltaW[k]); // Update the weight) ;
				}
				// Actualizamos el sesgo
				this->layers[i].neurons[j].w[0] -= (this->eta*this->layers[i].neurons[j].deltaW[0])
					- (this->mu * this->eta * this->layers[i].neurons[j].lastDeltaW[0]);
			}	
		}
	}

	// Modo offline
	else{
		for(int h=1; h<this->nOfLayers; h++){
			for(int j=1; j< this->layers[h].nOfNeurons; j++){
				for(int i=1; i<this->layers[h-1].nOfNeurons; i++){
					this->layers[h].neurons[j].w[i] -= (this->eta*this->layers[h].neurons[j].deltaW[i] / this->nOfTrainingPatterns)
					- (this->mu * (this->eta * this->layers[h].neurons[j].lastDeltaW[i])/this->nOfTrainingPatterns);
				}
				// Sesgo
				this->layers[h].neurons[j].w[0] -= (this->eta*this->layers[h].neurons[j].deltaW[0] / this->nOfTrainingPatterns)
					- (this->mu * (this->eta * this->layers[h].neurons[j].lastDeltaW[0])/this->nOfTrainingPatterns);
			}
		}
	}
	
	
	
}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork() { // anadido en practica1
	for(int i=1; i<this->nOfLayers; i++){
		std::cout << "Layer " << i << std::endl;
		for(int j=0; j<this->layers[i].nOfNeurons; j++){
			for(int k=0; k<this->layers[i].nOfNeurons + 1; k++){
				std::cout << this->layers[i].neurons[j].w[k] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << endl;
	}
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
// The step of adjusting the weights must be performed only in the online case
// If the algorithm is offline, the weightAdjustment must be performed in the "train" function
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::performEpoch(double* input, double* target, int errorFunction) {
	if(this->online == 1){
		for(int i=1; i<this->nOfLayers; i++){
			for(int j=0; j<this->layers[i].nOfNeurons; j++){
				for(int k=0; k<this->layers[i-1].nOfNeurons + 1; k++){
					this->layers[i].neurons[j].deltaW[k] = 0.0;
				}
			}
		}
	}

	this->feedInputs(input);
	this->forwardPropagate();
	this->backpropagateError(target, errorFunction);
	this->accumulateChange();

	if(this->online == 1){
		this->weightAdjustment();
	}
}

// ------------------------------
// Train the network for a dataset (one iteration of the external loop)
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::train(Dataset* trainDataset, int errorFunction) {
	if(this->online == 0){
		for(int i=1; i<this->nOfLayers; i++){
			for(int j=0; j<this->layers[i].nOfNeurons; j++){
				for(int k=0; k<this->layers[i-1].nOfNeurons + 1; k++){
					this->layers[i].neurons[j].deltaW[k] = 0.0;
				}
			}
		}
	}

	for(int i=0; i<trainDataset->nOfPatterns; i++){
		performEpoch(trainDataset->inputs[i], trainDataset->outputs[i],errorFunction);
	}
	

	if(this->online == 0){
		this->weightAdjustment();
	}
}

// ------------------------------
// Test the network with a dataset and return the error
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::test(Dataset* dataset, int errorFunction) {
	double sum = 0.0;

	for(int i=0; i<dataset->nOfPatterns; i++){
		this->feedInputs(dataset->inputs[i]);
		this->forwardPropagate();

		sum += this->obtainError(dataset->outputs[i], errorFunction);
	}

	if(errorFunction == 0)
		return sum / dataset->nOfPatterns; //MSE

	return -1 * (sum / dataset->nOfPatterns); //Cross Entropy
}


// ------------------------------
// Test the network with a dataset and return the CCR
double MultilayerPerceptron::testClassification(Dataset* dataset) {
	int ccr = 0.0;
	int expectedClass = 0, obtainedClass = 0;
	double *outArray = new double[this->layers[this->nOfLayers - 1].nOfNeurons];
	double maximo = 0.0, maximo2 = 0.0;

	for(int i=0; i<dataset->nOfPatterns; i++){
		this->feedInputs(dataset->inputs[i]);
		this->forwardPropagate();
		this->getOutputs(outArray);

		maximo = outArray[0];
		maximo2 = dataset->outputs[i][0];
		for(int j=1; j<dataset->nOfOutputs; j++){
			// Maximo de la clase obtenida
			if(maximo < outArray[j]){
				maximo = outArray[j];
				obtainedClass = j;
			}
			// maximo de la clase esperada
			if(maximo2 < dataset->outputs[i][j]){
				maximo2 = dataset->outputs[i][j];
				expectedClass = j;
			}
		}

		if(expectedClass == obtainedClass){
			ccr++;
		}

	}

	return ((double)ccr / dataset->nOfPatterns) * 100;
}


// ------------------------------
// Optional Kaggle: Obtain the predicted outputs for a dataset
void MultilayerPerceptron::predict(Dataset* dataset)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers-1].nOfNeurons;
	double * salidas = new double[numSalidas];
	
	cout << "Id,Category" << endl;
	
	for (i=0; i<dataset->nOfPatterns; i++){

		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		getOutputs(salidas);

		int maxIndex = 0;
		for (j = 0; j < numSalidas; j++)
			if (salidas[j] >= salidas[maxIndex])
				maxIndex = j;
		
		cout << i << "," << maxIndex << endl;

	}
}



// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
// Both training and test CCRs should be obtained and stored in ccrTrain and ccrTest
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::runBackPropagation(Dataset * trainDataset, Dataset * testDataset, int maxiter, double *errorTrain, double *errorTest, double *ccrTrain, double *ccrTest, int errorFunction)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	randomWeights();

	double minTrainError = 0;
	int iterWithoutImproving = 0;
	nOfTrainingPatterns = trainDataset->nOfPatterns;


	// Learning
	do {

		train(trainDataset,errorFunction);

		double trainError = test(trainDataset,errorFunction);
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

	if ( iterWithoutImproving!=50)
		restoreWeights();

	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();

	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for(int i=0; i<testDataset->nOfPatterns; i++){
		double* prediction = new double[testDataset->nOfOutputs];

		// Feed the inputs and propagate the values
		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for(int j=0; j<testDataset->nOfOutputs; j++)
			cout << testDataset->outputs[i][j] << " -- " << prediction[j] << " ";
		cout << endl;
		delete[] prediction;

	}

	*errorTest=test(testDataset,errorFunction);;
	*errorTrain=minTrainError;
	*ccrTest = testClassification(testDataset);
	*ccrTrain = testClassification(trainDataset);

}

// -------------------------
// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char * fileName)
{
	// Object for writing the file
	ofstream f(fileName);

	if(!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for(int i = 0; i < nOfLayers; i++)
	{
		f << " " << layers[i].nOfNeurons;
	}
	f << " " << outputFunction;
	f << endl;

	// Write the weight matrix of every layer
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				if(layers[i].neurons[j].w!=NULL)
				    f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;

}


// -----------------------
// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char * fileName)
{
	// Object for reading a file
	ifstream f(fileName);

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
	{
		f >> npl[i];
	}
	f >> outputFunction;

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				if(!(outputFunction==1 && (i==(nOfLayers-1)) && (k==(layers[i].nOfNeurons-1))))
					f >> layers[i].neurons[j].w[k];

	f.close();
	delete[] npl;

	return true;
}

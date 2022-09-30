//============================================================================
// Introduction to computational models
// Name        : la1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>    // To obtain current time time()
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <string.h>
#include <math.h>

#include "imc/MultilayerPerceptron.h"
#include "imc/util.h"


using namespace imc;
using namespace std;
using namespace util;

int main(int argc, char **argv) {
    // Process arguments of the command line
    bool mflag = 0,tflag = 0, Tflag = 0, wflag = 0, pflag = 0, iflag = 0, lflag = 0, hflag = 0, eflag=0, sflag=0;
    char *Tvalue = NULL, *wvalue = NULL, *tvalue = NULL;
    int c, ivalue = 0, lvalue = 0, hvalue = 0, svalue = 0, mvalue = 0, evalue = 0;

    opterr = 0;

    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "t:T:i:l:h:w:e:m:ps")) != -1)
    {
        // The parameters needed for using the optional prediction mode of Kaggle have been included.
        // You should add the rest of parameters needed for the lab assignment.
        switch(c){
            case 's': // Indica que los datos (train and test) se normalizan
                sflag = true;
                break;
            case 'm': // Valor del parametro mu
                mflag = true;
                mvalue = atoi(optarg); 
                break;
            case 'e': // Valor del parametro eta
                eflag = true;
                evalue = atoi(optarg);
                break;
            case 'h': // Numero de neuronas de cada capa oculta
                hflag = true;
                hvalue = atoi(optarg);
                break;
            case 'l': // Numero de capas ocultas del modelo de red neuronal
                lflag = true;
                lvalue = atoi(optarg);
                break;
            case 'i': // Numero de iteraciones del bucle externo
                iflag = true;
                ivalue = atoi(optarg);
                break;
            case 't': // Nombre del fichero con los datos de entrenamiento.
                tflag = true;
                tvalue = optarg;
                break;
            case 'T': // Nombre del fichero con los datos de test.
                Tflag = true;
                Tvalue = optarg;
                break;
            case 'w':
                wflag = true;
                wvalue = optarg;
                break;
            case 'p':
                pflag = true;
                break;
            case '?':
                if (optopt == 'T' || optopt == 'w' || optopt == 'p')
                    fprintf (stderr, "The option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                             "Unknown character `\\x%x'.\n",
                             optopt);
                return EXIT_FAILURE;
            default:
                return EXIT_FAILURE;
        }
    }

    if (!tflag){
        fprintf (stderr, "The option -t is required.\n");
        return EXIT_FAILURE;
    }

    if (!pflag) {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        // Multilayer perceptron object
    	MultilayerPerceptron mlp;

        // No se especifica i -> Numero de iteraciones es 1000

        if(!iflag){
            ivalue = 1000;
        }

        // No especifica T -> Se usan los datos de entrenamiento

        if(!Tflag){
            Tvalue = tvalue;
        }

        // No se especifica l -> Utilizar 1 capa oculta

        if(!lflag){
            lvalue = 1;
        }

        // No se especifica h -> Utilizar 5 neuronas por capa oculta

        if(!hflag){
            hvalue = 5;
        }

        // No se especifica e -> Utilizar e=0.1

        if(!eflag){
            evalue = 0.1;
        }

        // No se especifica m -> Utilizar m=0.9

        if(!mflag){
            mvalue = 0.9;
        }

        // Parameters of the mlp. For example, mlp.eta = value;
        mlp.eta = evalue;
        mlp.mu = mvalue;
    
        
    	int iterations = ivalue; // This should be corrected

        // Read training and test data: call to util::readData(...)
    	Dataset * trainDataset = util::readData(tvalue); // This should be corrected
    	Dataset * testDataset = util::readData(Tvalue); // This should be corrected

        // Initialize topology vector
    	int layers=lvalue; // This should be corrected
    	int * topology=new int[lvalue+2]; // This should be corrected
        topology[0] = trainDataset->nOfInputs; // Entrenamos la capa de entrada
        // Entrenamos las neuronas de las capas ocultas de la primera topologia
        for(int i=1; i<=layers; i++){
            topology[i] = hvalue;
        }
        // Entrenamos la capa de salida
        topology[layers+1] = trainDataset->nOfOutputs;
        // Initialize the network using the topology vector
        mlp.initialize(layers+2,topology);


        // Seed for random numbers
        int seeds[] = {1,2,3,4,5};
        double *testErrors = new double[5];
        double *trainErrors = new double[5];
        double bestTestError = 1;
        for(int i=0; i<5; i++){
            cout << "**********" << endl;
            cout << "SEED " << seeds[i] << endl;
            cout << "**********" << endl;
            srand(seeds[i]);
            mlp.runOnlineBackPropagation(trainDataset,testDataset,iterations,&(trainErrors[i]),&(testErrors[i]));
            cout << "We end!! => Final test error: " << testErrors[i] << endl;
            // Entrenamiento del error mediante back propagation
            // We save the weights every time we find a better model
            if(wflag && testErrors[i] <= bestTestError)
            {
                mlp.saveWeights(wvalue);
                bestTestError = testErrors[i];
            }
        }

        cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

        double averageTestError = 0, stdTestError = 0;
        double averageTrainError = 0, stdTrainError = 0;
        
        // Obtain training and test averages and standard deviations

        // Averrages

        for(int i=0; i<5; i++){
            averageTestError += testErrors[i];
            averageTrainError += trainErrors[i];
        }

        averageTestError = averageTestError / 5;
        averageTrainError = averageTrainError / 5;

        // Standard deviations

        for(int i=0; i<5; i++){
            stdTestError += pow(testErrors[i] - averageTestError, 2);
            stdTrainError += pow(trainErrors[i] - averageTrainError, 2);
        }

        stdTestError = sqrt(stdTestError / 5);
        stdTrainError = sqrt(stdTrainError / 5);

        cout << "FINAL REPORT" << endl;
        cout << "************" << endl;
        cout << "Train error (Mean +- SD): " << averageTrainError << " +- " << stdTrainError << endl;
        cout << "Test error (Mean +- SD):          " << averageTestError << " +- " << stdTestError << endl;
        return EXIT_SUCCESS;
    }
    else {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////
        
        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Initializing the network with the topology vector
        if(!wflag || !mlp.readWeights(wvalue))
        {
            cerr << "Error while reading weights, we can not continue" << endl;
            exit(-1);
        }

        // Reading training and test data: call to util::readData(...)
        Dataset *testDataset;
        testDataset = readData(Tvalue);
        if(testDataset == NULL)
        {
            cerr << "The test file is not valid, we can not continue" << endl;
            exit(-1);
        }

        mlp.predict(testDataset);

        return EXIT_SUCCESS;
    }

    
}


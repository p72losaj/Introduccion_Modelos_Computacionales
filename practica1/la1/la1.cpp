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
    bool Tflag = false, wflag = false, pflag = false;
    char *Tvalue = NULL, *wvalue = NULL;
    int c;

    opterr = 0;

    // TODO: Opciones por implementar

    bool tflag = false, iflag=false,lflag=false,hflag=false,eflag=false,mflag=false,sflag=false;
    char* tvalue=NULL;
    int ivalue=0,lvalue=0,hvalue=0;
    double evalue = 0.0, mvalue = 0.0;
    //////////////////////////////////////////

    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "t:T:i:l:h:e:m:w:ps")) != -1)
    {
        // The parameters needed for using the optional prediction mode of Kaggle have been included.
        // You should add the rest of parameters needed for the lab assignment.
        switch(c){
            // TODO: Case 't'
            case 't':
            tflag = true;
            tvalue = optarg;
            break;
            // TODO: Case 'i'
            case 'i':
            iflag = true;
            ivalue = atoi(optarg);
            break;
            // TODO: Case 'l'
            case 'l':
            lflag = true;
            lvalue = atoi(optarg);
            break;
            // TODO: Case 'h'
            case 'h':
            hflag = true;
            hvalue = atoi(optarg);
            break;
            // TODO: Case 'e'
            case 'e':
            eflag = true;
            evalue = atof(optarg);
            break;
            // TODO: Case 'm'
            case 'm':
            mflag = true;
            mvalue = atof(optarg);
            break;
            // TODO: Case 's'
            case 's':
            sflag = true;
            break;
            //////////////////////////////////////////
            case 'T':
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

    if(!tflag){
        cout << "Error: Missing option -t" << endl;
        return EXIT_FAILURE;
    }

    if (!pflag) {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        // Multilayer perceptron object
    	MultilayerPerceptron mlp;

        // Parameters of the mlp. For example, mlp.eta = value;
        if(!eflag){
            evalue = 0.1;
        }
        mlp.eta = evalue;
        if(!mflag){
            mvalue = 0.9;
        }
        mlp.mu = mvalue;
        if(!iflag){
            ivalue = 1000;
        } 
        int iterations = ivalue; // this should be correct
        // Read training and test data: call to util::readData(...)
        if(!Tflag){
            Tvalue = tvalue;
        }
        Dataset * trainDataset = util::readData(tvalue);
        Dataset * testDataset = util::readData(Tvalue);

        // Normalizamos los datos de entrenamiento
        if(sflag == true){
            // Escalar entrenamiento calculando minimo y maximo
            double * minTrain = util::minDatasetInputs(trainDataset); // minimo de cada columna de train
            double * maxTrain = util::maxDatasetInputs(trainDataset); // maximo de cada columna de train
            util::minMaxScalerDataSetInputs(trainDataset, -1.00, 1.00, minTrain, maxTrain); // normalizamos train
            util::minMaxScalerDataSetInputs(testDataset, -1.00, 1.00, minTrain, maxTrain); // normalizamos train
            // Escalar test calculando minimo y maximo
            double minTest = util::minDatasetOutputs(trainDataset); // minimo de cada columna de test
            double maxTest = util::maxDatasetOutputs(trainDataset); // maximo de cada columna de test
            util::minMaxScalerDataSetOutputs(trainDataset, 0.00, 1.00, minTest, maxTest); // normalizamos test
            util::minMaxScalerDataSetOutputs(testDataset, 0.00, 1.00, minTest, maxTest); // normalizamos test

        }

        // Initialize topology vector
        if(!lflag){
            lvalue = 1;
        }
    	int layers=lvalue; // This should be corrected
    	int * topology= new int[layers+2]; // This should be corrected
        if(!hflag){
            hvalue = 5;
        }
        topology[0] = trainDataset->nOfInputs; // Entrenamos las neuronas de entrada
        topology[layers+1] = trainDataset->nOfOutputs; // Entrenamos las neuronas de salida
        // Entrenamos las neuronas de la capa oculta
        for(int i=1; i<layers+1; i++){
            topology[i] = hvalue;
        }

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

            // We save the weights every time we find a better model
            if(wflag && testErrors[i] <= bestTestError)
            {
                mlp.saveWeights(wvalue);
                bestTestError = testErrors[i];
            }
        }

        cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

        double averageTestError = 0.0, stdTestError = 0.0;
        double averageTrainError = 0.0, stdTrainError = 0.0;
        
        // Obtain training and test averages and standard deviations

        for(int i=0; i<5; i++){
            averageTestError += testErrors[i];
            averageTrainError += trainErrors[i];
        }
        averageTestError /= 5;
        averageTrainError /= 5;

        for(int i=0; i<5; i++){
            stdTestError += pow(testErrors[i] - averageTestError, 2);
            stdTrainError += pow(trainErrors[i] - averageTrainError, 2);
        }
        stdTestError = sqrt(stdTestError/5);
        stdTrainError = sqrt(stdTrainError/5);


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


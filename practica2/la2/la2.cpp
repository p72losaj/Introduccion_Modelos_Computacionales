//============================================================================
// Introduction to computational models
// Name        : la2.cpp
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
#include <float.h>

#include "imc/MultilayerPerceptron.h"
#include "imc/util.h"


using namespace imc;
using namespace std;

int main(int argc, char **argv) {
	// Process the command line
    bool Tflag = 0, wflag = 0, pflag = 0;
    char *Tvalue = NULL, *wvalue = NULL;
    int c;

    // TODO: Opciones por implementar

    bool tflag = false, iflag=false,lflag=false,hflag=false,eflag=false,mflag=false,nflag=false;
    bool oflag = false, fflag = false, sflag = false;
    char* tvalue=NULL;
    int ivalue=0,lvalue=0,hvalue=0,fvalue=0, svalue = 0;
    double evalue = 0.0, mvalue = 0.0;

    opterr = 0;

    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "s::f:o::n::m:e:h:l:i:t:T:w:p")) != -1)
    {

        // The parameters needed for using the optional prediction mode of Kaggle have been included.
        // You should add the rest of parameters needed for the lab assignment.
        switch(c){
            case 's': // Funcion en la capa de salida
                sflag = true;
                svalue = 1;
                break;
            case 'f': // Funcion de error a emplear
                fflag = true;
                fvalue = atoi(optarg);
                break;
            case 'o': // Version a emplear
                oflag = true;
                break;
            // Anadido practica1
            case 't': // Dataset de entrenamiento
            tflag = true;
            tvalue = optarg;
            break;
            // Anadido practica1
            case 'i': // Numero maximo de iteraciones
            iflag = true;
            ivalue = atoi(optarg);
            break;
            // Anadido en practica1
            case 'l': // Numero de capas ocultas
            lflag = true;
            lvalue = atoi(optarg);
            break;
            // Anadido en practica1
            case 'h': // Numero de neuronas por capa oculta
            hflag = true;
            hvalue = atoi(optarg);
            break;
            // Anadido en practica1
            case 'e': // Parametro eta
            eflag = true;
            evalue = atof(optarg);
            break;
            // Anadido en practica1
            case 'm': // Parametro mu
            mflag = true;
            mvalue = atof(optarg);
            break;
            // Anadido en practica2
            case 'n': // Normalizacion de los datos de entrada
            nflag = true;
            break;
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
                if (optopt == 'f' || optopt == 'l' || optopt=='m' || optopt=='e' || optopt == 'h' || optopt == 'T' || optopt == 'w' || optopt == 'p' || optopt == 't')
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
        cerr << "The option -t is mandatory" << endl;
        return EXIT_FAILURE;
    }


    if (!pflag) {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        // Multilayer perceptron object
    	MultilayerPerceptron mlp;

        // Parameters of the mlp. For example, mlp.eta = value

        if(!eflag) evalue = 0.7;
        
        mlp.eta = evalue;

        if(!mflag) mvalue = 1;
        
        mlp.mu = mvalue;

        if(!iflag) ivalue = 1000;

        mlp.online = oflag;

        mlp.outputFunction = sflag;

    	// Type of error considered
    	int error=fflag; // This should be completed

    	// Maximum number of iterations
    	int maxIter=ivalue; // This should be completed

        // Read training and test data: call to util::readData(...)
        if(!Tflag) Tvalue = tvalue;
        
    	Dataset * trainDataset = util::readData(tvalue); // This should be corrected
    	Dataset * testDataset = util::readData(Tvalue); // This should be corrected

        // Normalizamos los datos de entrenamiento
        if(nflag == true){
            // Escalar entrenamiento calculando minimo y maximo
            double * minTrain = util::minDatasetInputs(trainDataset); // minimo de cada columna de train
            double * maxTrain = util::maxDatasetInputs(trainDataset); // maximo de cada columna de train
            util::minMaxScalerDataSetInputs(trainDataset, -1.00, 1.00, minTrain, maxTrain); // normalizamos train
            util::minMaxScalerDataSetInputs(testDataset, -1.00, 1.00, minTrain, maxTrain); // normalizamos train
        }

        // Initialize topology vector
        if(!lflag) lvalue = 1;
        
        int layers=lvalue;

        int *topology = new int[layers+2];
        
        if(!hflag) hvalue = 5;
        
        topology[0] = trainDataset->nOfInputs;
        
        // Entrenamos las neuronas de la capa oculta
        for(int i=1; i<layers+1; i++){
            topology[i] = hvalue;
        }

        topology[layers+2-1] = trainDataset->nOfOutputs;
        
        mlp.initialize(layers+2,topology);

		// Seed for random numbers
		int seeds[] = {1,2,3,4,5};
		double *trainErrors = new double[5];
		double *testErrors = new double[5];
		double *trainCCRs = new double[5];
		double *testCCRs = new double[5];
		double bestTestError = DBL_MAX;
		for(int i=0; i<5; i++){
			cout << "**********" << endl;
			cout << "SEED " << seeds[i] << endl;
			cout << "**********" << endl;
			srand(seeds[i]);
			mlp.runBackPropagation(trainDataset,testDataset,maxIter,&(trainErrors[i]),&(testErrors[i]),&(trainCCRs[i]),&(testCCRs[i]),error);
			cout << "We end!! => Final test CCR: " << testCCRs[i] << endl;

			// We save the weights every time we find a better model
			if(wflag && testErrors[i] <= bestTestError)
			{
				mlp.saveWeights(wvalue);
				bestTestError = testErrors[i];
			}
		}


		double trainAverageError = 0, trainStdError = 0;
		double testAverageError = 0, testStdError = 0;
		double trainAverageCCR = 0, trainStdCCR = 0;
		double testAverageCCR = 0, testStdCCR = 0;

        // Obtain training and test averages and standard deviations or error

        // Anadido en practica1
        for(int i=0; i<5; i++){
            testAverageError += testErrors[i];
            trainAverageError += trainErrors[i];
        }
        testAverageError /= 5;
        trainAverageError /= 5;

        for(int i=0; i<5; i++){
            testStdError += pow(testErrors[i] - testAverageError, 2);
            trainStdError += pow(trainErrors[i] - trainAverageError, 2);
        }
        testStdError = sqrt(testStdError/5);
        trainStdError = sqrt(trainStdError/5);

        // Obtain training and test average and standard deviations of CCR

        for(int i=0; i<5; i++){
            testAverageCCR += testCCRs[i];
            trainAverageCCR += trainCCRs[i];
        }
        testAverageCCR /= 5;
        trainAverageCCR /= 5;

        for(int i=0; i<5; i++){
            testStdCCR += pow(testCCRs[i] - testAverageCCR, 2);
            trainStdCCR += pow(trainCCRs[i] - trainAverageCCR, 2);
        }
        testStdCCR = sqrt(testStdCCR/5);
        trainStdCCR = sqrt(trainStdCCR/5);

		cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

		cout << "FINAL REPORT" << endl;
		cout << "*************" << endl;
	    cout << "Train error (Mean +- SD): " << trainAverageError << " +- " << trainStdError << endl;
	    cout << "Test error (Mean +- SD): " << testAverageError << " +- " << testStdError << endl;
	    cout << "Train CCR (Mean +- SD): " << trainAverageCCR << " +- " << trainStdCCR << endl;
	    cout << "Test CCR (Mean +- SD): " << testAverageCCR << " +- " << testStdCCR << endl;
		return EXIT_SUCCESS;
    } else {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////

        // You do not have to modify anything from here.
        
        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Initializing the network with the topology vector
        if(!wflag || !mlp.readWeights(wvalue))
        {
            cerr << "Error while reading weights, we can not continue" << endl;
            exit(-1);
        }

        // Reading training and test data: call to readData(...)
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


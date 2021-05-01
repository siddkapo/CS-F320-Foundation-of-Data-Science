#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <random>

#define LEARNING_RATE 1
#define EPOCHS 1000
#define NEURON_NUM 10

/*
* Class labels belong to the range [0, M - 1]
* Output vector is 1-hot vector
* A = 0 is Sigmoid, A = 1 is ReLU
* Input File format :
* N
* M
* L
* testL
* H
* A
* filename
* *********
* Inside filename.txt
* 
* x11 x12 x13 ... x1N class1
* x21 x22 x23 ... x2N class2
*  .   .   .  ...  .    .
*  .   .   .  ...  .    .
* xL1 xL2 xL3 ... xLN classL
* x11 x12 x13 ... x1N class1
* x21 x22 x23 ... x2N class2
*  .   .   .  ...  .    .
*  .   .   .  ...  .    .
* xtestL1 xtestL2 xtestL3 ... xtestLN classtestL
* 
*/

double SigmoidActivation(double x) {
	double sigmoid = std::exp(-x) + 1;
	sigmoid = 1 / sigmoid;
	return sigmoid;
}

void SoftmaxActivation(double* input, double* output, int N) {
	double denominator = 0.0;
	for(int i = 0; i < N; ++i) {
		denominator += std::exp(input[i]);
	}
	for(int i = 0; i < N; ++i) {
		output[i] = std::exp(input[i]) / denominator;
	}
}

double ReLUActivation(double x) {
	return std::max(x, 0.0);
}

double WeightedSum(double* input, double* weights, int N) {
	double wsum = 0.0;
	for(int i = 0; i <= N; ++i) {
		wsum += input[i] * weights[i];
	}
	return wsum;
}

double LossFunction(double* outputLayer, int dataclass) {
	double loss = outputLayer[dataclass];
	loss = -std::log(loss);
	return loss;
}

void NormalizeDataset(double** dataset, int N, int L) {
	double mean[N];
	double variance[N];
	double min[N];
	double max[N];

	// Calculating min and max
	for(int i = 0; i < L; ++i) {
		for(int j = 0; j < N; ++j) {
			if(i == 0) {
				min[j] = dataset[i][j];
				max[j] = dataset[i][j];
			} else {
				if(dataset[i][j] < min[j]) min[j] = dataset[i][j];
				if(dataset[i][j] > max[j]) max[j] = dataset[i][j];
			}
		}
	}

	// Scaling input dataset
	for(int i = 0; i < L; ++i) {
		for(int j = 0; j < N; ++j) {
			dataset[i][j] = (dataset[i][j] - min[j]) / (max[j] - min[j]);
		}
	}

	// // Calculating mean for each feature
	// for(int i = 0; i < L; ++i) {
	// 	for(int j = 0; j < N; ++j) {
	// 		if(i == 0) {
	// 			mean[j] = dataset[i][j];
	// 			variance[j] = 0.0;
	// 		}
	// 		else mean[j] += dataset[i][j];
	// 		if(i == L - 1) mean[j] /= L;
	// 	}
	// }

	// // Calculating variance for each feature
	// for(int i = 0; i < L; ++i) {
	// 	for(int j = 0; j < N; ++j) {
	// 		variance[j] += (dataset[i][j] - mean[j]) * (dataset[i][j] - mean[j]);
	// 		if(i == L) variance[j] /= L - 1;
	// 	}
	// }

	// // Normalizing dataset
	// for(int i = 0; i < L; ++i) {
	// 	for(int j = 0; j < N; ++j) {
	// 		dataset[i][j] = (dataset[i][j] - mean[j]) / variance[j];
	// 	}
	// }
}

void InitWeights(double** weightSetIn, double** weightSetOut, double*** weightSetHid, int N, int M, int H, int A) {
	// Initialize weights to random values

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_real_distribution<double> distribution(0.0, 1.0);

	double lower;
	double upper;
	double random;
	
	if(A == 0) {
		lower = -std::sqrt(6.0 / ((double) (M + N + 1))); // Normalized Xavier Weight Initialization
		upper = std::sqrt(6.0 / ((double) (M + N + 1))); // Normalized Xavier Weight Initialization
	} else if(A == 1) {
		lower = 0.0; // He Weight Initialization
		upper = std::sqrt(2.0 / ((double) N + 1)); // He Weight Initialization
	}
	// std::cout << "weightSetIn\n";
	int range = (H == 0) ? M : NEURON_NUM;
	for(int i = 0; i < range; ++i) {
		for(int j = 0; j <= N; ++j) {;
			random = distribution(generator);
			weightSetIn[i][j] = lower + (upper - lower) * random;
			// std::cout << weightSetIn[i][j] << " ";
		}
		// std::cout << "\n";
	}

	// std::cout << "weightSetOut\n";
	if(A == 0) {
		lower = -std::sqrt(6.0 / ((double) (M + NEURON_NUM + 1))); // Normalized Xavier Weight Initialization
		upper = std::sqrt(6.0 / ((double) (M + NEURON_NUM + 1))); // Normalized Xavier Weight Initialization
	} else if(A == 1) {
		lower = 0.0; // He Weight Initialization
		upper = std::sqrt(2.0 / ((double) NEURON_NUM + 1)); // He Weight Initialization
	}
	if(weightSetOut != NULL) {
		for(int i = 0; i < M; ++i) {
			for(int j = 0; j <= NEURON_NUM; ++j) {
				random = distribution(generator);
				weightSetOut[i][j] = lower + (upper - lower) * random;
				// std::cout << weightSetIn[i][j] << " ";
			}
			// std::cout << "\n";
		}
	}
	
	// std::cout << "weightSetHid\n";
	if(A == 0) {
		lower = -std::sqrt(6.0 / ((double) (NEURON_NUM + NEURON_NUM + 1))); // Normalized Xavier Weight Initialization
		upper = std::sqrt(6.0 / ((double) (NEURON_NUM + NEURON_NUM + 1))); // Normalized Xavier Weight Initialization
	} else if(A == 1) {
		lower = 0.0; // He Weight Initialization
		upper = std::sqrt(2.0 / ((double) NEURON_NUM + 1)); // He Weight Initialization
	}
	if(weightSetHid != NULL) {
		for(int i = 0; i < H - 1; ++i) {
			for(int j = 0; j < NEURON_NUM; ++j) {
				for(int k = 0; k <= NEURON_NUM; ++k) {
					random = distribution(generator);
					weightSetHid[i][j][k] = lower + (upper - lower) * random;
					// std::cout << weightSetHid[i][j][k] << " ";
				}
				// std::cout << "\t";
			}
			// std::cout << "\n";
		}
	}
}

void FeedForward(double* inputLayer, double** weightSetIn, double** hiddenLayers, double*** weightSetHid, double* outputLayer, double** weightSetOut, int N, int M, int H, int A) {

	if(H == 0) {
		for(int i = 0; i < M; ++i) {
			outputLayer[i] = WeightedSum(inputLayer, weightSetIn[i], N);
		}
	} else {
		for(int i = 0; i < NEURON_NUM; ++i) {
			if(A == 0) hiddenLayers[0][i] = SigmoidActivation(WeightedSum(inputLayer, weightSetIn[i], N));
			else if(A == 1) hiddenLayers[0][i] = ReLUActivation(WeightedSum(inputLayer, weightSetIn[i], N));
		}
		for(int i = 1; i < H; ++i) {
			for(int j = 0; j < NEURON_NUM; ++j) {
				if(A == 0) hiddenLayers[i][j] = SigmoidActivation(WeightedSum(hiddenLayers[i - 1], weightSetHid[i - 1][j], NEURON_NUM));
				else if(A == 1) hiddenLayers[i][j] = ReLUActivation(WeightedSum(hiddenLayers[i - 1], weightSetHid[i - 1][j], NEURON_NUM));
			}
		}
		for(int i = 0; i < M; ++i) {
			outputLayer[i] = WeightedSum(hiddenLayers[H - 1], weightSetOut[i], NEURON_NUM);
		}
	}
	SoftmaxActivation(outputLayer, outputLayer, M);
}

void CalculateGradient(double* inputLayer, double** weightSetIn, double** gradientIn, double** hiddenLayers, double*** weightSetHid, double*** gradientHid, double* outputLayer, double** weightSetOut, double** gradientOut, int dataclass, int N, int M, int H, int A) {
	// Calculating the negative gradient of each weight w.r.t the cross entropy loss function

	double* deltaOut = new double [M];
	double** deltaHid = new double* [H];
	for(int i = 0; i < H; ++i) deltaHid[i] = new double [NEURON_NUM + 1];

	if(H == 0) {
		for(int i = 0; i < M; ++i) {
			// Gradient of Input Weight Set
			if(i == dataclass) deltaOut[i] = 1 - outputLayer[i];
			else deltaOut[i] = -outputLayer[i];
			for(int j = 0; j <= N; ++j) gradientIn[i][j] = deltaOut[i] * inputLayer[j];
		}
	} else {
		// Gradient and Delta of Output Weight Set
		for(int i = 0; i < M; ++i) {
			if(i == dataclass) deltaOut[i] = 1 - outputLayer[i];
			else deltaOut[i] = -outputLayer[i];
			for(int j = 0; j <= NEURON_NUM; ++j) gradientOut[i][j] = deltaOut[i] * hiddenLayers[H - 1][j];
		}

		// Gradient of Hidden Weight Sets
		double tmp;
		for(int i = H - 1; i >= 0; --i) {
			for(int j = 0; j <= NEURON_NUM; ++j) {
				tmp = 0.0;
				if(i == H - 1) {
					for(int k = 0; k < M; ++k) {
						tmp += weightSetOut[k][j] * deltaOut[k];
					}
				} else {
					for(int k = 0; k < NEURON_NUM; ++k) {
						tmp += weightSetHid[i][k][j] * deltaHid[i + 1][k];
					}
				}
				if(A == 0) tmp *= hiddenLayers[i][j] * (1 - hiddenLayers[i][j]);
				else if(A == 1) tmp *= (hiddenLayers[i][j] > 0) ? 1 : 0;
				deltaHid[i][j] = tmp;
			}
			if(i < H - 1) {
				for(int j = 0; j < NEURON_NUM; ++j) {
					for(int k = 0; k <= NEURON_NUM; ++k) {
						gradientHid[i][j][k] = deltaHid[i + 1][j] * hiddenLayers[i][k];
					}
				}
			}
		}

		// Gradient of Input Weight Set
		for(int i = 0; i < NEURON_NUM; ++i) {
			for(int j = 0; j <= N; ++j) {
				gradientIn[i][j] = deltaHid[0][i] * inputLayer[j];
			}
		}
	}
}

void UpdateWeights(double** weightSetIn, double** gradientIn, double*** weightSetHid, double*** gradientHid, double** weightSetOut, double** gradientOut, int N, int M, int H) {
	// Updating weights using the calculated negative gradients of all the weights

	if(H == 0) {
		for(int i = 0; i < M; ++i) {
			for(int j = 0; j <= N; ++j) {
				weightSetIn[i][j] += LEARNING_RATE * gradientIn[i][j];
			}
		}
	} else {
		for(int i = 0; i < NEURON_NUM; ++i) {
			for(int j = 0; j <= N; ++j) {
				weightSetIn[i][j] += LEARNING_RATE * gradientIn[i][j];
			}
		}
		for(int i = 0; i < H - 1; ++i) {
			for(int j = 0; j < NEURON_NUM; ++j) {
				for(int k = 0; k <= NEURON_NUM; ++k) {
					weightSetHid[i][j][k] += LEARNING_RATE * gradientHid[i][j][k];
				}
			}
		}
		for(int i = 0; i < M; ++i) {
			for(int j = 0; j <= NEURON_NUM; ++j) {
				weightSetOut[i][j] += LEARNING_RATE * gradientOut[i][j];
			}
		}
	}
}

void BackPropogation(double* inputLayer, double** weightSetIn, double** hiddenLayers, double*** weightSetHid, double * outputLayer, double** weightSetOut, int dataclass, int N, int M, int H, int A) {

	double** gradientIn; // Gradient of weights of each neuron of 1st layer to the next hidden layer
	double*** gradientHid; // Gradient of weights of each neuron of some hidden layer to the next hidden layer
	double** gradientOut; // Gradient of weights of each neuron of last hidden layer to the output layer

	if(H == 0) {
		gradientIn = new double* [M];
		for(int i = 0; i < M; ++i) gradientIn[i] = new double [N + 1];

		gradientOut = NULL;
		gradientHid = NULL;
	} else {
		gradientIn = new double* [NEURON_NUM];
		for(int i = 0; i < NEURON_NUM; ++i) gradientIn[i] = new double [N + 1];
		
		gradientOut = new double* [M];
		for(int i = 0; i < M; ++i) gradientOut[i] = new double [NEURON_NUM + 1];

		gradientHid = new double** [H - 1];
		for(int i = 0; i < H - 1; ++i) gradientHid[i] = new double* [NEURON_NUM];
		for(int i = 0; i < H - 1; ++i) {
			for(int j = 0; j < NEURON_NUM; ++j) gradientHid[i][j] = new double [NEURON_NUM + 1];
		}
	}

	CalculateGradient(inputLayer, weightSetIn, gradientIn, hiddenLayers, weightSetHid, gradientHid, outputLayer, weightSetOut, gradientOut, dataclass, N, M, H, A);

	UpdateWeights(weightSetIn, gradientIn, weightSetHid, gradientHid, weightSetOut, gradientOut, N, M, H);
}

int main() {

	int N; // Number of dimesions of input
	int M; // Number of output classes
	int L; // Length of dataset
	int testL; // Length of Test Set
	int A; // For taking choice of activation function
	int H; // Number of hidden layers
	std::string filename; // Filename of test and training data
	std::fstream myfile; // For reading the file
	
	std::cin >> N >> M >> L >> testL >> H >> A >> filename;

	if(A != 0 && A != 1) {
		std::cout << "INCORRECT ACTIVATION FUNCTION CHOICE\n";
		return 0;
	}

	myfile.open(filename);

	double** dataset = new double* [L]; // Reads and stores dataset
	for(int i = 0; i < L; ++i) dataset[i] = new double [N];
	int* dataclass = new int [L]; // Class of the data
	
	double** testset = new double* [testL]; // Reads and stores testset
	for(int i = 0; i < testL; ++i) testset[i] = new double [N];
	int* testclass = new int [testL]; // Class of the test data
	
	for(int i = 0; i < L + testL; ++i) { // Each line of the dataset is of the form : x1 x2 x3 ... xN class
		for(int j = 0; j < N; ++j) {
			if(i < L) {
				myfile >> dataset[i][j];
			} else {
				myfile >> testset[i - L][j];
			}
		}
		if(i < L) myfile >> dataclass[i];
		else myfile >> testclass[i - L];
	}
	NormalizeDataset(dataset, N, L);
	NormalizeDataset(testset, N, testL);

	myfile.close();

	// Setting up the Neural Network
	double* inputLayer;
	inputLayer = new double [N + 1]; // Input Layer
	inputLayer[N] = 1.0; // For Bias
	double* outputLayer;
	outputLayer = new double [M]; // Final output layer

	double** hiddenLayers; // H * (NEURON_NUM + 1)
	double** weightSetIn; // NEURON_NUM * (N + 1) or M * (N + 1)
	double** weightSetOut; // M * (NEURON_NUM + 1)
	double*** weightSetHid; // (H - 1) * NEURON_NUM * (NEURON_NUM + 1)

	if(H > 0) {
		weightSetIn = new double* [NEURON_NUM]; // Weights of each neuron of 1st layer to the next hidden layer
		for(int i = 0; i < NEURON_NUM; ++i) weightSetIn[i] = new double [N + 1];
		
		weightSetOut = new double* [M]; // Weights of each neuron of last hidden layer to the output layer
		for(int i = 0; i < M; ++i) weightSetOut[i] = new double [NEURON_NUM + 1];
		
		hiddenLayers = new double* [H]; // Neurons of each hidden layer
		for(int i = 0; i < H; ++i) {
			hiddenLayers[i] = new double [NEURON_NUM + 1];
			hiddenLayers[i][NEURON_NUM] = 1.0; // For Bias
		}

		weightSetHid = new double** [H - 1]; // Weights of each neuron of each hidden layer to the next hidden layer
		for(int i = 0; i < H - 1; ++i) weightSetHid[i] = new double* [NEURON_NUM];
		for(int i = 0; i < H - 1; ++i) {
			for(int j = 0; j < NEURON_NUM; ++j) weightSetHid[i][j] = new double [NEURON_NUM + 1];
		}
	} else {
		weightSetIn = new double* [M]; // Weights of each neuron of 1st layer to the next hidden layer
		for(int i = 0; i < M; ++i) weightSetIn[i] = new double [N + 1];

		weightSetOut = NULL;
		weightSetHid = NULL;
		hiddenLayers = NULL;
	}


	// Training the Neural Network
	InitWeights(weightSetIn, weightSetOut, weightSetHid, N, M, H, A);

	double loss;
	double totalLoss;

	for(int epoch = 1; epoch <= EPOCHS; ++epoch) {
		
		totalLoss = 0.0;
		for(int i = 0; i < L; ++i) {

			for(int j = 0; j < N; ++j) inputLayer[j] = dataset[i][j];

			FeedForward(inputLayer, weightSetIn, hiddenLayers, weightSetHid, outputLayer, weightSetOut, N, M, H, A);
			
			loss = LossFunction(outputLayer, dataclass[i]);
			totalLoss += loss;

			BackPropogation(inputLayer, weightSetIn, hiddenLayers, weightSetHid, outputLayer, weightSetOut, dataclass[i], N, M, H, A);

			// for(int j = 0; j < M; ++j) {
			// 	std::cout << outputLayer[j] << " ";
			// }
			// std::cout << "\n";
		}

		std::cout << "Total Loss for Epoch " << epoch << " = " << totalLoss << "\n";
	}

	// Testing the Trained Neural Network
	int predictedClass;
	double accuracy = 0.0;
	double maxProb;
	for(int i = 0; i < testL; ++i) {
		maxProb = 0.0;
		for(int j = 0; j < N; ++j) inputLayer[j] = testset[i][j];
		FeedForward(inputLayer, weightSetIn, hiddenLayers, weightSetHid, outputLayer, weightSetOut, N, M, H, A);
		for(int j = 0; j < M; ++j) {
			if(outputLayer[j] > maxProb) {
				maxProb = outputLayer[j];
				predictedClass = j;
			}
			std::cout << outputLayer[j] << " ";
		}
		if(predictedClass == testclass[i]) accuracy++;
		std::cout << "\nPredicted Class = " << predictedClass << "\tActual Class = " << testclass[i] << "\n";
	}
	accuracy /= (double) testL;
	accuracy *= 100;
	std::cout << "Accuracy = " << accuracy << "%\n";

	return 0;
}
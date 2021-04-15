#include <iostream>
#include <cmath>
#include <algorithm>
#include <array>
#include <utility>
#include <cstdlib>

void MatrixInvert(double** arr, double** inv, int n) {
	
	double** augarr = new double* [n];
	for(int i = 0; i < n; ++i) {
		augarr[i] = new double [2 * n];
	}
	for(int i = 0; i < n; ++i) {
		for(int j = 0; j < 2 * n; ++j) {
			if(j < n) augarr[i][j] = arr[i][j];
			else if(j - n == i) augarr[i][j] = 1.0;
			else augarr[i][j] = 0.0;
		}
	}
	for(int i = 0; i < n; ++i) {
		double temp = augarr[i][i];
		for(int j = 0; j < n; ++j) {
			augarr[i][j] /= temp;
			augarr[i][j + n] /= temp;
		}
		for(int j = i + 1; j < n; ++j) {
			temp = augarr[j][i];
			for(int k = 0; k < n; ++k) {
				augarr[j][k] -= augarr[i][k] * temp;
				augarr[j][k + n] -= augarr[i][k + n] * temp;
			}
		}
	}
	for(int i = n - 1; i >= 0; --i) {
		double temp = augarr[i][i];
		for(int j = n - 1; j >= 0; --j) {
			augarr[i][j] /= temp;
			augarr[i][j + n] /= temp;
		}
		for(int j = i - 1; j >= 0; --j) {
			temp = augarr[j][i];
			for(int k = n - 1; k >= 0; --k) {
				augarr[j][k] -= augarr[i][k] * temp;
				augarr[j][k + n] -= augarr[i][k + n] * temp;
			}
		}
	}
	for(int i = 0; i < n; ++i) {
		for(int j = 0; j < 2 * n; ++j) {
			if(j >= n) inv[i][j - n] = augarr[i][j];
		}
	}
}

void ParameterCalculator(double** phi, int n, int m, double dataset[][2], double** parameters) {
	
	// std::cout << "Phi\n";
	// for(int i = 0; i < n; ++i) {
	// 	for(int j = 0; j <= m; ++j) {
	// 		std::cout << "\t" << phi[i][j];
	// 	}
	// 	std::cout << "\n";
	// }
	// std::cout << "\n";
	
	// Calculating X_arr = phi(T) * phi
	// std::cout << "Phi(T) * Phi\n";
	double** xarr;
	xarr = new double* [m + 1];
	for(int i = 0; i <= m; ++i) {
		xarr[i] = new double [m + 1];
	}
	for(int i = 0; i <= m; ++i) {
		for(int j = 0; j <= m; ++j) {
			xarr[i][j] = 0.0;
			for(int k = 0; k < n; ++k) {
				xarr[i][j] += phi[k][i] * phi[k][j];
			}
			// std::cout << "\t" << xarr[i][j];
		}
		// std::cout << "\n";
	}
	// std::cout << "\n";
	
	// Calculating X_arr inverse
	// std::cout << "inverse(Phi(T) * Phi)\n";
	double** invx;
	invx = new double* [m + 1];
	for(int i = 0; i <= m; ++i) {
		invx[i] = new double [m + 1];
	}
	MatrixInvert(xarr, invx, m + 1);
	for(int i = 0; i <= m; ++i) {
		for(int j = 0; j <= m; ++j) {
			// std::cout << "\t" << invx[i][j];
		}
		// std::cout << "\n";
	}
	// std::cout << "\n";
	
	// Calculating Y_arr = Inv(X_Arr) * Phi(T)
	// std::cout << "inverse(Phi(T) * Phi) * Phi(T)\n";
	double yarr[m + 1][n];
	for(int i = 0; i <= m; ++i) {
		for(int j = 0; j < n; ++j) {
			yarr[i][j] = 0.0;
			for(int k = 0; k <= m; ++k) {
				yarr[i][j] += invx[i][k] * phi[j][k];
			}
			// std::cout << "\t" << yarr[i][j];
		}
		// std::cout << "\n";
	}
	// std::cout << "\n";
	
	// Calculating parameters = Y_arr * Y_coord
	std::cout << "Parameters = Inv(Phi(T) * Phi) * Phi(T) * Y\n";
	for(int i = 0; i <= m; ++i) {
		parameters[m][i] = 0.0;
		for(int j = 0; j < n; ++j) {
			parameters[m][i] += yarr[i][j] * dataset[j][1];
		}
		std::cout << "\t" << parameters[m][i];
	}
	std::cout << "\n\n";
}

std::pair<int, std::pair<double*, double**>> PolynomialRegression(double dataset[][2], int n) {
	// Basis Functions of the form ==> {x^0, x^1, ......, x^m}

	int maxDegree = n; // Number of parameters equal to number of data points
	
	double** phi = new double* [n];
	for(int i = 0; i < n; ++i) {
		phi[i] = new double [maxDegree + 1];
	}
	for(int i = 0; i < n; ++i) {
		for(int j = 0; j <= maxDegree; ++j) {
			if(j == 0) phi[i][j] = 1.0;
			else phi[i][j] = dataset[i][0] * phi[i][j - 1];
		}
	}

	double** parameters;
	parameters = new double* [maxDegree + 1];
	for(int i = 0; i <= maxDegree; ++i) {
		parameters[i] = new double [maxDegree + 1];
	}
	for(int i = 0; i <= maxDegree; ++i) {
		for(int j = 0; j <= maxDegree; ++j) {
			parameters[i][j] = 0.0; // All parameters initialized to 0
		}
	}

	double* rmse;
	rmse = new double[maxDegree + 1];

	// Iterating over each polynomial degree
	for(int m = 0; m <= maxDegree; ++m) {
		std::cout << "******************************Basis Functions******************************\n";
		for(int i = 0; i <= m; ++i) {
			if(i == 0) std::cout << "\t1";
			else if(i == 1) std::cout << "\tx";
			else std::cout << "\tx^" << i;
		}
		std::cout << "\n\n";

		ParameterCalculator(phi, n, m, dataset, parameters);

		// Calculating RMS Error
		rmse[m] = 0.0;
		for(int i = 0; i < n; ++i) {
			double fy = 0.0;
			for(int j = 0; j <= m; ++j) {
				fy += parameters[m][j] * std::pow(dataset[i][0], j);
			}
			rmse[m] += (fy - dataset[i][1]) * (fy - dataset[i][1]);
		}
		rmse[m] /= n;
		rmse[m] = std::sqrt(rmse[m]);
		std::cout << "\tRMS Error = " << rmse[m] << "\n";
	}

	return std::make_pair(maxDegree + 1, std::make_pair(rmse, parameters));
}

std::pair<int, std::pair<double*, double**>> GaussianRegression(double dataset[][2], int n) {
	// Basis Functions of the form ==> e^(-(x - mean_i)^2 / 2s^2). s will be assumed to be equal to 1

	int maxBells = n; // Number of curves upto to number of data points
	double mean = 0.0;
	
	double** phi = new double* [n];
	for(int i = 0; i < n; ++i) {
		phi[i] = new double [maxBells];
	}

	double** parameters;
	parameters = new double* [maxBells];
	for(int i = 0; i < maxBells; ++i) {
		parameters[i] = new double [maxBells];
	}
	for(int i = 0; i < maxBells; ++i) {
		for(int j = 0; j < maxBells; ++j) {
			parameters[i][j] = 0.0; // All parameters initialized to 0
		}
	}

	double* rmse;
	rmse = new double[maxBells + 1];

	// Iterating over number of gaussian basis functions
	for(int m = 0; m < maxBells; ++m) {
		std::cout << "******************************Basis Functions******************************\n";
		mean = dataset[0][0] + dataset[n - 1][0];
		mean /= m + 2;
		for(int i = 0; i <= m; ++i) {
			std::cout << "\tMu_" << i << " = " << (i + 1.0) * mean;
		}
		std::cout << "\n\n";

		for(int i = 0; i < n; ++i) {
			for(int j = 0; j <= m; ++j) {
				double temp = dataset[i][0] - (j + 1.0) * mean;
				temp *= temp;
				temp /= 2.0;
				temp *= -1.0;
				phi[i][j] = std::exp(temp);
			}
		}

		ParameterCalculator(phi, n, m, dataset, parameters);

		// Calculating RMS Error
		rmse[m] = 0.0;
		for(int i = 0; i < n; ++i) {
			double fy = 0.0;
			for(int j = 0; j <= m; ++j) {
				double temp = dataset[i][0] - (j + 1.0) * mean;
				temp *= temp;
				temp /= 2.0;
				temp *= -1.0;
				fy += parameters[m][j] * std::exp(temp);
			}
			rmse[m] += (fy - dataset[i][1]) * (fy - dataset[i][1]);
		}
		rmse[m] /= n;
		rmse[m] = std::sqrt(rmse[m]);
		std::cout << "\tRMS Error = " << rmse[m] << "\n";
	}

	return std::make_pair(maxBells, std::make_pair(rmse, parameters));
}

std::pair<int, std::pair<double*, double**>> FourierRegression(double dataset[][2], int n) {
	// Basis Functions of the form ==> 1 sin(nwx) cos(nwx). Assume w = 0.5

	int maxDegree = n / 2; // Number of parameters equal to number of data points
	if(maxDegree % 2 == 0) maxDegree++;
	if(n % 4 == 0) maxDegree--;
	
	double** phi = new double* [n];
	for(int i = 0; i < n; ++i) {
		phi[i] = new double [maxDegree];
	}

	double** parameters;
	parameters = new double* [n];
	for(int i = 0; i < n; ++i) {
		parameters[i] = new double [n];
	}
	for(int i = 0; i < n; ++i) {
		for(int j = 0; j < n; ++j) {
			parameters[i][j] = 0.0; // All parameters initialized to 0
		}
	}

	double* rmse;
	rmse = new double[maxDegree];

	// Iterating over each sine cosine period
	for(int m = 0; m < maxDegree; ++m) {
		std::cout << "******************************Basis Functions******************************\n";
		for(int i = 0; i <= m; ++i) {
			if(i == 0) std::cout << "\t1";
			else if(i == 1) std::cout << "\tsin(x/2)";
			else std::cout << "\tsin(" << i << "x/2)";
		}
		for(int i = 1; i <= m; ++i) {
			if(i == 1) std::cout << "\tcos(x/2)";
			else std::cout << "\tcos(" << i << "x/2)";
		}
		std::cout << "\n\n";

		for(int i = 0; i < n; ++i) {
			for(int j = 0; j <= m; j++) {
				double temp = dataset[i][0] * 0.5 * j;
				if(j == 0) phi[i][j] = 1.0;
				else {
					phi[i][j] = std::sin(temp);
					phi[i][j + m] = std::cos(temp);
				}
			}
		}

		ParameterCalculator(phi, n, 2 * m, dataset, parameters);

		// Calculating RMS Error
		rmse[m] = 0.0;
		for(int i = 0; i < n; ++i) {
			double fy = parameters[2 * m][0];
			// std::cout << "fy = " << fy << "\n";
			for(int j = 1; j <= m; ++j) {
				double temp = dataset[i][0] * 0.5 * j;
				// std::cout << "\tsin(" << temp << ") = " << std::sin(temp) << "\n"; 
				fy += parameters[2 * m][j] * std::sin(temp);
			}
			for(int j = m + 1; j <= 2 * m; ++j) {
				double temp = dataset[i][0] * 0.5 * (j - m);
				// std::cout << "\tcos(" << temp << ") = " << std::cos(temp) << "\n";
				fy += parameters[2 * m][j] * std::cos(temp);
			}
			rmse[m] += (fy - dataset[i][1]) * (fy - dataset[i][1]);
		}
		rmse[m] /= n;
		rmse[m] = std::sqrt(rmse[m]);
		std::cout << "\tRMS Error = " << rmse[m] << "\n";
	}

	return std::make_pair(maxDegree, std::make_pair(rmse, parameters));
}

int main() {

	int n; // Length of dataset
	std::cin >> n;
	double dataset[n][2]; // Assuming 2-dimensional dataset
	for(int i = 0; i < n; ++i) {
		std::cin >> dataset[i][0] >> dataset[i][1]; // input format per line : x-coordinate;y-coordinate
	}

	std::cout << "Select Basis Function Type :\n";
	std::cout << "\t1 - Polynomial\n";
	std::cout << "\t2 - Gaussian\n";
	std::cout << "\t3 - Fourier\n";
	int choice;
	std::cin >> choice; // Enter Choice of Basis Functions

	std::pair<int, std::pair<double*, double**>> params; // No. of parameter list, RMS Error for each case, List of Coefficients

	switch(choice) {
		case 1:
			params = PolynomialRegression(dataset, n);
			break;
		case 2:
			params = GaussianRegression(dataset, n);
			break;
		case 3:
			params = FourierRegression(dataset, n);
			break;
		default:
			std::cout << "Incorrect Selection\n";
	}

	return 0;
}
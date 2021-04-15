To Compile
$ g++ linearregression.cpp -o lr.out

To Run
$ ./lr.out < input.txt > output.txt

Input File Format
	Line 1 must have a number N
	Next N lines 2 space separated values representing X & Y coordinates
	Last line must be 1 for Polynomial Regression / 2 for Gaussian / 3 for Fourier

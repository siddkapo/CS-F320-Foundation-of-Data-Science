The assignment consists of writing a program to implement different basis functions for linear regression.
1. Polynomial - Done
2. Sigmoidal - Skipped
3. Gaussian - Done
4. Fourier - Done
5. Splines (including B-splines) - Skipped
6. Wavelets - Skipped

To Compile
$ g++ linearregression.cpp -o lr.out

To Run
$ ./lr.out < input.txt > output.txt

Input File Format
	Line 1 must have a number N
	Next N lines 2 space separated values representing X & Y coordinates
	Last line must be 1 for Polynomial Regression / 2 for Gaussian / 3 for Fourier

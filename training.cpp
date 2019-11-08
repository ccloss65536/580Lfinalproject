#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

using namespace std;

//take training image file name

//take training labels

//num training samples

//Image size dimensions(width,heigh)

//num input neurons
//num hidden neurons
//num output neurons
//epochs?
//learning rate
//momentum
//epsilon

//input layer to hidden layer

//input hidden later to output layer

//Output layer

//Image

//filestream to read data

//allocate memorry

//ELU Function Kevin Yan
double ELU(double x) {
	if(x > 0) 
		return x;
	else 
		return learningRate*(exp(x)-1);
}
//ELU derivative Function Kevin Yan
double dELU(double x) {
	if(x > 0)
		return 1;
	else
		return ELU(x) + learningRate;
//perceptron

//Back Propogation Algo

//Learning

//Reading input

//save weights

//main

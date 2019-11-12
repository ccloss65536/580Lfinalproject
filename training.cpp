#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <thread>
#include <Eigen/Dense> 

using namespace std;
using namespace Eigen;
class NeuralNetwork{
//take training image file name

//take training labels

//num training samples



int num_input_neurons;
//num hidden neurons
int num_hidden_neurons;
//num output neurons
int num_output_neurons;
//epochs?
//learning rate
double learning_rate;
vector<MatrixXf> layers;

//epochs?
//learning rate Carl
//momentum
//epsilon Carl

//input layer to hidden layer Carl

//input hidden layer(s) to output layer Carl

//Output layer Carl

//Image

//filestream to read data

//allocate memory Carl

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

//Back Propogation Algo Carl

//Learning Carl

//Reading input

//save weights
};
//main

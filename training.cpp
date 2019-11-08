#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

using namespace std;

//take training image file name

//take training labels

//num training samples

//Image size dimensi`ons(width,heigh)

//num input neurons
int num_input_neurons;
//num hidden neurons
int num_hidden_neurons;
//num output neurons
int num_output_neurons;
//epochs?
//learning rate
double learning_rate;
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
//perceptron Kevin Yan
void perceptron() {
	//set hidden nodes to 0
	for(int i = 0; i <= num_input_nodes; i++) {
		hidden_nodes = 0;
	}
}
//Back Propogation Algo

//Learning

//Reading input

//save weights

//main

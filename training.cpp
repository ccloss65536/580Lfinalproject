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
	vector<MatrixXd> layers;

	//epochs?
	//learning rate Carl
	//momentum??
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
	}
	 

	//perceptron 
	
	//This returns the loss for a single example, preconverted into a vector
	double loss(const& VectorXf input_vector, const& Vector reference_vec){
		VectorXd temp = input_vector;
		for(MatrixXd l : layers){
			temp = l * temp;
			for(int i = 0; i < temp.size(); i++){
				temp[i] = ELU(temp[i]);
			}
		return (reference_vec - temp).dot(reference_vec - temp);
	}


	//Learning Carl
	//My plan is to calculate loss on every training example in parallel, to add to a variable, then backpropagate.

	//Reading input

	//save weights
};
//main

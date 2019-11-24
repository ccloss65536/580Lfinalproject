#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <thread>
#include <mutex>
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
	mutex loss_sum_lock, producer, consumer, file; //Do these need to be pointers to mutexes? probably not
	vector<pair<VectorXd>> vecs_to_calc; //put input and expected output vectors into this, and take them out to process
	unsigned int pro_buffer_index, con_buffer_index, buffer_size, con_training_total, pro_training_total;
	 
	 


	//epochs?
	//epsilon Carl

	//input layer to hidden layer Carl

	//input hidden layer(s) to output layer Carl

	//Output layer Carl

	//Image

	//filestream to read data

	//allocate memory 
	pair<VectorXd> generate_training_example(istream& file){
		//make sure to grab and ungrab the file mutex as necessary to avoid corruption 
	}
	
	void producer_thread(istream& file, unsigned int training_size){
		while(file){
			pair<VectorXd> example = generate_training_example(file);
			auto l = lock_guard<mutex>(producer); //take the mutex until the lock_guard leaves scope
			vecs_to_calc[pro_buffer_index] = example;
			buffer_index = (pro_buffer_index + 1) % buffer_size;
		}
	}

	void consumer_thread(double& total_loss, unsigned int training_size){
		while(training_total < training_size){
			auto l = unique_lock<mutex>(consumer);
			l.lock();
			training_total++;
			pair<VectorXd> example = vecs_to_calc[con_buffer_index]
			con_buffer_index = (con_buffer_index + 1) % buffer_size;
			l.unlock();
			double loss = loss(example.first, example.second);
			l = unique_lock(loss_sum_lock);
			l.lock();
			total_loss += loss;
			l.unlock();	
		}
	}



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
	double loss(const VectorXd& input_vector, const VectorXd& reference_vec){
		VectorXd temp = input_vector;
		for(MatrixXd l : layers){
			temp = l * temp;
			for(int i = 0; i < temp.size(); i++){
				temp[i] = ELU(temp[i]);
			}
		return (reference_vec - temp).dot(reference_vec - temp); //feel free to change this to something faster
	}


	//Learning Carl
	//My plan is to calculate loss on every training example in parallel, to add to a variable, then backpropagate (I am unclaimimng this part).

	void train(const& string filename){
		//open file here
		double loss = 0;
		ifstream f(filename);
		vector<thread> threads;
		//loop to initialize threads with either producer args or consumer args
		//joins
		//backpropagate

	}
		





	//Reading input

	//save weights
};
//main

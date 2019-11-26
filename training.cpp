#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <random>
#include <Eigen/Dense> 
#include <cmath>

using namespace std;
using namespace Eigen;


class NeuralNetwork{
public:
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
	condition_variable full, empty;
	vector<pair<VectorXd>> vecs_to_calc; //put input and expected output vectors into this, and take them out to process
	unsigned int pro_buffer_index, con_buffer_index, buffer_size, con_training_total, pro_training_total, buffer_taken;
	unsigned int num_producers, num_consumers;
	
	NeuralNetwork(double learning_rate, num_layers/*...*/){
		this->learning_rate = learning_rate;
		pro_buffer_index = con_buffer_index = con_training_total = buffer_taken = 0;
		//initalize vector buffer

		//TODO: finish this

	}
		

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
		while(file && pro_training_total < training_size){
			pair<VectorXd> example = generate_training_example(file);
			auto l = unique_lock(producer);
			while(buffer_taken >= buffer_size) full.wait(l);
			vecs_to_calc[pro_buffer_index] = example;
			pro_buffer_index = (pro_buffer_index + 1) % buffer_size;
			buffer_taken++;
			empty.notify_one();
			pro_training_total++;
			producer.unlock();
		}
	}

	void consumer_thread(double& total_loss, unsigned int training_size){
		while(training_total < training_size){
			auto l = unique_lock(consumer); //lock is taken as soon as this object is constructed
			while(buffer_taken <= 0){
				if(file_ended){
					l.unlock();
					return;
				}
				empty.wait(l);
			}

			training_total++;
			pair<VectorXd> example = vecs_to_calc[con_buffer_index]
			con_buffer_index = (con_buffer_index + 1) % buffer_size;
			buffer_taken--;
			full.notify_one();
			l.unlock();
			double loss = loss(example.first, example.second);
			l = unique_lock(loss_sum_lock);
			total_loss += loss;
			l.unlock();	
		}
	}



	//ELU Function Kevin Yan
	double ELU(double x) {
		if(x > 0) 
			return x;
		else 
			return learning_rate*(exp(x)-1);
	}
	//ELU derivative Function Kevin Yan
	double dELU(double x) {
		if(x > 0)
			return 1;
		else
			return ELU(x) + learning_rate;
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


	//Learning
	//My plan is to calculate loss on every training example in parallel, to add to a variable, then backpropagate (I am unclaimimng this part).

	void train(const& string filename, unsigned int training_size){
		double loss = 0;
		ifstream f(filename);
		vector<thread> threads;
		for(unsigned int i = 0; i < num_producers; i++){
			threads.push_back(thread(producer_thread, this, f, training_size));
		}
		for(unsigned int i = 0; i < num_consumers; i++){
			threads.push_back(thread(consumer_thread, this, loss, training_size));
		}
		for(thread& t : threads){
			t.join();
		}
		
		
		//backpropagate

	}
		





	//Reading input

	//save weights
};
//main

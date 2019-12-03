#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <thread>
#include <mutex>
#include <vector>
#include <condition_variable>
#include "Eigen/Dense"
#include <cmath>

using namespace std;
using namespace Eigen;



const int IMAGE_ROWS = 28;
const int IMAGE_COLS = 28;
const int IMAGE_SIZE =  IMAGE_ROWS * IMAGE_COLS; //28*28, the number of pixels in a MNIST image


//WHYYY are MNIST numbers in big endian
int read_num(char* buff, size_t size){
	int out = 5;
	char* test_p = (char*)&out;
	if(*test_p){ //system is little endian, make sure to reverse the number
		for(size_t i = size - 1; i <= 0; i--){
			*test_p = *(buff + i);
			*test_p++;
		}
	}
	else{
		for(size_t i = 0; i < size; i++){
			*test_p = *(buff + i);
			*test_p++;
		}
	}
	return out;
}




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
	int epochs;
	//learning rate
	double learning_rate;
	vector<MatrixXd> layers;
	vector<RowVectorXd> layer_sums;
	mutex loss_sum_lock, producer, consumer, generate, layer_s;
	condition_variable full, empty;
	vector<pair<RowVectorXd,RowVectorXd>> vecs_to_calc; //put input and expected output vectors into this, and take them out to process

	unsigned int pro_buffer_index, con_buffer_index, buffer_size, con_training_total, pro_training_total, buffer_taken;
	unsigned int num_producers, num_consumers;
	
	NeuralNetwork(double learning_rate, int num_layers, int epochs, int hidden_layer_size, int buffer_size){
		num_input_neurons = IMAGE_SIZE;
		num_output_neurons = 10;
		num_hidden_neurons = hidden_layer_size;
		this->learning_rate = learning_rate;
		this->buffer_size = buffer_size;
		pro_buffer_index = con_buffer_index = con_training_total = buffer_taken = 0;
		vecs_to_calc = vector(buffer_size);
		/* We use an extra hidden "neuron" to perpetuate the bias term, the
		   first element of each set of inputs
		*/
		layers.emplace(layers.end(), MatrixXd::Random(num_input_neurons + 1, hidden_layer_size + 1));
		for(int i = 0; i  < num_layers - 2; i++){
			layers.emplace(layers.end(), MatrixXd::Random(hidden_layer_size + 1, hidden_layer_size + 1));
		}
		layers.emplace(layers.end(), MatrixXd::Random(hidden_layer_size + 1, num_output_neurons + 1));

		for(MatrixXd& l : layers){
			l(0,0) = 1;
			for(int i = 1; i < l.rows(); i++) l(i, 0) = 0;
		}
		layer_sums = vector<RowVectorXd>(num_layers);
		for(unsigned int i = 0; i < num_layers; i++){
			layer_sums[i] = RowVectorXd::Zero(layers[i].cols);
		}
		this->epochs = epochs;

				

		//TODO: finish this

	}

	//Image

	pair<RowVectorXd,RowVectorXd> generate_training_example(istream& images, istream& labels){
		//TODO: finish this
		char buff[IMAGE_SIZE];
		char label;
		auto l = unique_lock<mutex>(generate);
		images.read(buff, IMAGE_SIZE);
		labels.read(&label, 1)
		l.unlock();
		RowVectorXd image(IMAGE_SIZE + 1);
		image[0] = 1;
		for(int i = 1; i < IMAGE_SIZE + 1; i++){
			image[i] = buff[i];
		}
		RowVectorXd target = RowVectorXd::Zero(num_output_neurons + 1);
		target[label + 1] = 1;
		target[0] = 1; //bias
		return pair<RowVectorXd,RowVectorXd>(image,target);


	}
	
	void producer_thread(istream& images, istream& labels, unsigned int training_size){
		while(images && pro_training_total < training_size){
			pair<RowVectorXd,RowVectorXd> example = generate_training_example(images, labels);
			auto l = unique_lock<mutex>(producer);
			while(buffer_taken >= buffer_size) full.wait(l);
			vecs_to_calc[pro_buffer_index] = example;
			pro_buffer_index = (pro_buffer_index + 1) % buffer_size;
			buffer_taken++;
			empty.notify_one();
			pro_training_total++;
			producer.unlock();
		}
	}

	void consumer_thread(double& total_loss, unsigned int training_size, RowVectorXd& diffs){
		while(training_total < training_size){
			auto l = unique_lock<mutex>(consumer); //lock is taken as soon as this object is constructed
			while(buffer_taken <= 0){
				empty.wait(l);
			}

			training_total++;
			pair<RowVectorXd,RowVectorXd> example = vecs_to_calc[con_buffer_index]
			con_buffer_index = (con_buffer_index + 1) % buffer_size;
			buffer_taken--;
			full.notify_one();
			l.unlock();
			RowVectorXd last_inputs(num_hidden_neurons);
			double loss = loss(example.first, example.second, last_inputs);
			l = unique_lock<mutex>(loss_sum_lock);
			total_loss += loss;
			diffs += example.second - example.first;
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
	//Always pass Eigen matrices & vectors by reference!
	//We use true gradient descent since it is easiy parallelizable.
	double loss(const RowVectorXd& input_vector, const RowVectorXd& reference_vec, RowVectorXd& last_input ){
		RowVectorXd temp = input_vector;
		int i = 0;
		for(MatrixXd l : layers){
			auto lock = unique_lock<mutex>(layer_s);
			layer_sums[i] += temp;
			lock.unlock();
			temp = temp * l;
			for(int i = 0; i < temp.size(); i++){
				temp[i] = ELU(temp[i]);
			}
			i++;
		}
		return pow( (reference_vec - temp).sum(), 2); //feel free to change this to something faster
	}


	//Learning
	//My plan is to calculate loss on every training example in parallel, to add to a variable, then backpropagate.

	void train(const string& image_file, const string& label_file){
		for(int i = 0; i < epochs; i++){
			double loss = 0;
			RowVectorXd sum_last_inputs = RowVectorXd::Zero(num_hidden_neurons);
			RowVectorXd diffs = RowVectorXd::Zero(num_output_neurons);
			ifstream images(image_file);
			ifstream labels(label_file);
			char magic_buff[4];
			images.read(magic_buff, 4);
			int magic = read_num(magic_buff, 4);

			if(magic != 2051){
				cerr << "Bad image data! Magic: " << magic << endl;
				exit(magic);
			}
			labels.read(magic_buff, 4);
			magic = read_num(magic_buff, 4);
			if(magic != 2049){
				cerr << "Bad label data! Magic:" << magic << endl;
				exit(magic);
			}

			char num_examples_buff[4] = {0,0,0,0};
			labels.read(num_examples_buff, 4);
			int training_size = read_num(num_examples_buff, 4);
			
			images.seek(ios_base::cur, 4*3); //skip over the number of examples, row size, and column size in the image data

			vector<thread> threads;
			for(unsigned int i = 0; i < num_producers; i++){
				threads.push_back(thread(producer_thread, this, i, training_size));
			}
			for(unsigned int i = 0; i < num_consumers; i++){
				threads.push_back(thread(consumer_thread, this, loss, training_size));
			}
			for(thread& t : threads){
				t.join();
			} 
			
			final_layer = layers.back();
			RowVectorXd d_diffs(final_layer.cols());
			for(int j = 0; j < final_layer.cols(); j++) d_diffs[j] = dELU(diffs[j]);
			RowVectorXd sigma_v = (diffs.array() * d_diffs.array() * sum_last_inputs.array()).matrix() //array allows for simple component-wise ops

			for(unsigned int i = layers.size() - 2; i >= 0; i--){
				//find next sigma vector, then add learning * sigma * layer_sum to each col
				RowVectorXd sigma_v_next(layers[i - 1].cols());
				for(int j = 0; j<  sigma_v_next.cols(); j++){
					sigma_v_next[j] = layers[i].row(j).dot(sigma_v) * dELU(layer_sums[i][j]);
				}
				for(int j = 1; j < layers[i + 1].cols();j++){
					layers[i + 1].col(j) += learning_rate * (sigma_v.array() * layer_sums[i+1].array()).matrix();
				}
				sigma_v = sigma_v_next;
			}


			}
			
			f.close();
		}
		

	}

	//save weights
};
//main

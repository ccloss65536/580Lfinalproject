#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <thread>
#include <vector>
#include <stdint.h>
#include <condition_variable>
#include "Eigen/Dense"
#include <cmath>

using namespace std;
using namespace Eigen;


const string TRAINING_IMAGES_FILENAME = "baby_mnist/train-images-idx3-ubyte";
const string TRAINING_LABELS_FILENAME = "baby_mnist/train-labels-idx1-ubyte";
const string TESTING_IMAGES_FILENAME = "baby_mnist/t10k-images-idx3-ubyte";
const string TESTING_LABELS_FILENAME = "baby_mnist/t10k-labels-idx1-ubyte";
const int IMAGE_ROWS = 28;
const int IMAGE_COLS = 28;
const int IMAGE_SIZE =  IMAGE_ROWS * IMAGE_COLS; //28*28, the number of pixels in a MNIST image


//WHYYY are MNIST numbers in big endian
//refactored to not require an external buffer, since that is annoying
//Use this to read in the 4-byte numbers from MNIST files or the numbers may be backwards!!!
int read_num(istream& in, int size){
	if(size > 8 || size <= 0) {
		cerr << "Bad size for number!";
		exit(67);
	}
	uint8_t buff[8] = {0,0,0,0,0,0,0,0};
	in.read( (char*)buff, size);
	int out = 5;
	uint8_t* test_p = (uint8_t*)&out;
	if(*test_p){ //system is little endian, make sure to reverse the number
		for(int i = size - 1; i >= 0; i--){
			*test_p = *(buff + i);
			test_p++;
		}
	}
	else{
		for(int i = 0; i < size; i++){
			*test_p = *(buff + i);
			test_p++;
		}
	}
	return out;
}




class NeuralNetwork{
public:
	//input neuron;
	double *out1;
	//hidden neurons
	double **hidden_out;
	double **hidden_in;
	//output neurons;
	double *output_out;
	double *output_in;


	//take training image file name

	//take training labels

	//num training samples
	//num layer
	int num_layers;
	//num hidden layers
	int num_hidden_layers;
  //num input neurons
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
	vector<pair<RowVectorXd,RowVectorXd>> vecs_to_calc; //put input and expected output vectors into this, and take them out to process
	bool is_training;

	NeuralNetwork(double learning_rate, int num_layers, int epochs, int hidden_layer_size){
		num_input_neurons = IMAGE_SIZE;
		num_output_neurons = 10;
		num_hidden_neurons = hidden_layer_size;
		this->learning_rate = learning_rate;
		this->num_layers = num_layers;
		num_hidden_layers = num_layers - 2;
		/* We use an extra hidden "neuron" to perpetuate the bias term, the
		   first element of each set of inputsoutput
		*/
		if(num_layers == 2){
			layers.emplace_back(MatrixXd::Random(num_input_neurons + 1, num_output_neurons + 1));
		}
		else{
			layers.emplace(layers.end(), MatrixXd::Random(num_input_neurons + 1, hidden_layer_size + 1));
			for(int i = 0; i  < num_layers - 2; i++){
				layers.emplace(layers.end(), MatrixXd::Random(hidden_layer_size + 1, hidden_layer_size + 1));
			}
			layers.emplace(layers.end(), MatrixXd::Random(hidden_layer_size + 1, num_output_neurons + 1));
		}
		for(MatrixXd& l : layers){
			l(0,0) = 1;
			for(int i = 1; i < l.rows(); i++) l(i, 0) = 0;
		}
		layer_sums = vector<RowVectorXd>(num_layers);
		for(int i = 0; i < num_layers; i++){
			layer_sums[i] = RowVectorXd::Zero(layers[i].rows());
		}
		this->epochs = epochs;
		is_training = false;

		//Allocate space for incoming and outgoing values of each neuron layer for testing (Kevin)

		out1 = new double[num_input_neurons + 1];
		hidden_out = new double*[num_layers-2];	//allocate layers-2 double pointers
		for(int i = 0; i < num_layers-2; i++) {
			hidden_out[i] = new double[num_hidden_neurons + 1]; //allocate space for each layer + 1 bias
		}
		hidden_in = new double*[num_layers-2];	//allocate layers-2 double pointers
		for(int i = 0; i < num_layers-2; i++) {
			hidden_in[i] = new double[num_hidden_neurons + 1]; //allocate space for each layer + 1 bias
		}
		output_in = new double[num_output_neurons + 1];
		output_out = new double[num_output_neurons + 1];


	}

	//Image

	pair<RowVectorXd,RowVectorXd> generate_training_example(istream& images, istream& labels){
		uint8_t buff[IMAGE_SIZE];
		char label;
		images.read( (char*)buff, IMAGE_SIZE);
		labels.read( &label, 1);
		RowVectorXd image(IMAGE_SIZE + 1);
		image[0] = 1;
		for(int i = 0; i < IMAGE_SIZE; i++){
			image[i + 1] = buff[i];
		}
		RowVectorXd target = RowVectorXd::Zero(num_output_neurons + 1);
		target[label + 1] = 1;
		target[0] = 1; //bias
		return pair<RowVectorXd,RowVectorXd>(image,target);


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

	RowVectorXd evaluate(const RowVectorXd& input_vector, const RowVectorXd& reference_vec){
		RowVectorXd temp = input_vector;
		int i = 0;
		for(MatrixXd l : layers){
			if(is_training) layer_sums[i] = temp;
			temp = temp * l;
			for(int j = 0; j < temp.cols(); j++){
				temp[j] = ELU(temp[j]);
			}
			i++;
		}
		return temp;

	}
	//This returns the loss for a single example, given the precomputed result and the correct answer
	//Always pass Eigen matrices & vectors by reference!
	double loss(const RowVectorXd& input_vector, const RowVectorXd& reference_vec){
		return pow( (reference_vec - input_vector).sum(), 2); //feel free to change this to something faster
	}


	//Learning
	//My plan is to calculate loss on every training example in parallel, to add to a variable, then backpropagate.


	void train(const string& image_file, const string& label_file){
		for(int i = 0; i < epochs; i++){
			is_training = true;
			ifstream images(image_file);
			ifstream labels(label_file);
			int magic = read_num(images, 4);

			if(magic != 2051){
				cerr << "Bad image data! Magic: " << magic << endl;
				exit(magic);
			}
			magic = read_num(labels, 4);
			if(magic != 2049){
				cerr << "Bad label data! Magic:" << magic << endl;
				exit(magic);
			}

			int training_size = read_num(labels, 4);
			//cout << labels.gcount() << endl;

			images.seekg(4*3, ios_base::cur); //skip over the number of examples, row size, and column size in the image data
			//find next sigma vector, then add learning * sigma * layer_sum to each col
			double prev_loss = 9999999999999999;
			double loss_ex = 0;
			for(int k = 0; k < training_size && prev_loss - loss_ex < .001; k++){
				cout << prev_loss - loss_ex << endl;
				pair<RowVectorXd,RowVectorXd> example = generate_training_example(images, labels);
				RowVectorXd result = evaluate(example.first ,example.second);
				prev_loss = loss_ex;
				loss_ex = loss(result, example.second);

				RowVectorXd diff = example.second - result;
				MatrixXd& final_layer = layers.back();
				RowVectorXd d_diff(final_layer.cols());
				//cout << diff.cols() << endl;
				for(int j = 0; j < final_layer.cols(); j++) d_diff[j] = dELU(diff[j]);
				RowVectorXd sigma_v = (diff.array() * d_diff.array()).matrix(); //array allows for simple component-wise ops

				for( int m = layers.size() - 1; m > 0; m--){

					RowVectorXd sigma_v_next(layers[i - 1].cols());
					for(int j = 0; j<  sigma_v_next.cols(); j++){

						sigma_v_next[j] = layers[m].row(j).dot(sigma_v) * dELU(layer_sums[m][j]);
					}
					for(int j = 1; j < layers[m].cols();j++){

						auto temp = learning_rate * sigma_v[j] * layer_sums[m];
						layers[m].col(j) += temp;
					}
					sigma_v = sigma_v_next;
				}
		}
		images.close();
		labels.close();
	}


	}
	//calculate outputs giving input
	void perceptron() {
			//set all hidden nodes to 0;
			for(int i = 0; i < num_hidden_layers; i++) {
				for(int j = 1; j < num_hidden_neurons+1; j++) {
					hidden_in[i][j] = 0.0;
				}
			}
			//set
			for(int i = 1; i < num_output_neurons+1; i++) {
				output_in[i] = 0.0;
			}
			//input layer to hidden layer
			for(int i = 1; i < num_input_neurons+1; i++) {
				for(int j = 1; j < num_hidden_neurons+1; j++) {
					hidden_in[0][j] += out1[i]*layers[0](i,j);
				}
			}
			//do all hidden layers
			for(int i = 1; i < num_hidden_layers; i++) {
				for(int j = 1; j < num_hidden_neurons+1; j++) {
					for(int k = 2 ; k < num_hidden_neurons+1; k++) {
						hidden_in[i][k] += hidden_out[i-1][j]*layers[i](j,k);
					}
				}
				//Run ELU on incoming to get outgoing
				for(int j = 1; j < num_hidden_neurons+1; j++) {
					hidden_out[i][j] = ELU(hidden_in[i][j]);
				}
			}
			//final layer
			for(int i = 1; i < num_hidden_neurons+1; i++) {
				for(int j = 1; j < num_output_neurons+1; j++) {
					output_in[j] += hidden_out[num_hidden_layers-1][i]*layers[num_layers-1].coeff(i,j);
				}
			}

	}
	//method to test (Kevin Yan)
	void testing(string testing_images_filename, string testing_labels_filename) {
		ifstream testing_images(testing_images_filename);
		ifstream testing_labels(testing_labels_filename);

		// //read binary image and label files
		// testing_images.open(testing_images_filename,ios::binary);
		// testing_labels.open(testing_labels_filename,ios::binary);
		//use read_num to get header info
		int image_magic_num = read_num(testing_images, 4);
		int num_images = read_num(testing_images,4);
		int num_rows = read_num(testing_images,4);
		int num_cols = read_num(testing_images,4);
		int label_magic_num = read_num(testing_labels, 4);
		int num_labels = read_num(testing_labels,4);
		char buffer;
		char label;
		int image_matrix[IMAGE_ROWS][IMAGE_COLS];
		int correctCount = 0;

		//check magic numbers
		if(image_magic_num != 2051) {
			cerr << "Bad Testing Image Data! " << image_magic_num << endl;
			exit(image_magic_num);
		}
		if(label_magic_num != 2049){
			cerr << "Bad Testing label data! " << label_magic_num << endl;
			exit(label_magic_num);
		}
		//loop through images in test set
		for(int img_index = 0; img_index < num_images; img_index++) {
			//read image data
			//put into matrix and input neurons of nn
			int position = 1;
			for(int i = 0; i < IMAGE_ROWS; i++) {
				for(int j = 0; j < IMAGE_COLS; j++) {
					testing_images.read(&buffer, sizeof(char));
					image_matrix[i][j] = buffer;	//might not need this line since we put it into an array and don't need the matrix really
					out1[position] = buffer;
					position++;
				}
			}

			//get label of testing example
			testing_labels.read(&label, sizeof(char));
			//run inputs through nn
			perceptron();
			//get nn prediction
			int prediction = 0;
			for(int i = 1; i < num_output_neurons; i++) {
				if(output_out[i] > output_out[prediction]) {
					prediction = i;
				}
			}
			if(prediction == label) {
				correctCount++;
			}
		}
		cout << "Correct Count: " << correctCount << endl;
		cout << "Total Count: " << num_images << endl;
	}

	//save weights Carl?
};


int main(int argc, char** argv){
	double learning_rate = (argc < 2)? .1 : stod(string(argv[1]));
	int num_layers = (argc < 3)? 3: stoi(argv[2]);
	int epochs = (argc < 4)? 10: stoi(argv[3]);
	int hidden_layer_size = (argc < 5)? 11: stoi(argv[4]);
	int buffer_size = (argc < 6)? 50: stoi(argv[5]);
	NeuralNetwork net(learning_rate, num_layers, epochs, hidden_layer_size);
	net.train(TRAINING_IMAGES_FILENAME, TRAINING_LABELS_FILENAME);
	net.testing(TESTING_IMAGES_FILENAME, TESTING_LABELS_FILENAME); //the nn param is presumably goimg to be removed

}

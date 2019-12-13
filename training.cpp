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


const string TRAINING_IMAGES_FILENAME = "mnist/train-images-idx3-ubyte";
const string TRAINING_LABELS_FILENAME = "mnist/train-labels-idx1-ubyte";
const string TESTING_IMAGES_FILENAME = "baby_mnist/t10k-images-idx3-ubyte";
const string TESTING_LABELS_FILENAME = "baby_mnist/t10k-labels-idx1-ubyte";
const int IMAGE_ROWS = 28;
const int IMAGE_COLS = 28;
const int IMAGE_SIZE =  IMAGE_ROWS * IMAGE_COLS; //28*28, the number of pixels in a MNIST image
const int BATCH_SIZE = 60;


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
	double momentum_constant;
	double elu_weight;
	vector<MatrixXd> layers;
	vector<RowVectorXd> layer_inputs;
	vector<pair<RowVectorXd,RowVectorXd>> vecs_to_calc; //put input and expected output vectors into this, and take them out to process
	bool is_training;
	int example_num;

	NeuralNetwork(double learning_rate, int num_layers, int epochs, int hidden_layer_size, double momentum, double elu_weight){
		num_input_neurons = IMAGE_SIZE;
		num_output_neurons = 10;
		num_hidden_neurons = hidden_layer_size;
		this->learning_rate = learning_rate;
		this->num_layers = num_layers;
		this->elu_weight = elu_weight;
		momentum_constant = momentum;
		num_hidden_layers = num_layers - 2;

		/* We use an extra hidden "neuron" to perpetuate the bias term, the
		   first element of each set of inputsoutput
		*/
		if(num_layers == 2){
			layers.emplace_back(MatrixXd::Random(num_input_neurons + 1, num_output_neurons + 1));
		}
		else{
			layers.emplace(layers.end(), MatrixXd::Random(num_input_neurons + 1, hidden_layer_size + 1));
			for(int i = 0; i  < num_layers - 3; i++){
				layers.emplace(layers.end(), MatrixXd::Random(hidden_layer_size + 1, hidden_layer_size + 1));
			}
			layers.emplace(layers.end(), MatrixXd::Random(hidden_layer_size + 1, num_output_neurons + 1));
		}
		for(MatrixXd& l : layers){
			l = (l.array() * .00005).matrix(); //reducing the absolute value of the weights makes the network actually descend
			l(0,0) = 1;
			for(int i = 1; i < l.rows(); i++) l(i, 0) = 0;
		}
		layer_inputs = vector<RowVectorXd>(num_layers);
		for(int i = 0; i < num_layers - 1; i++){
			layer_inputs[i] = RowVectorXd::Zero(layers[i].rows());
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
		example_num = 0;


	}

	//Image

	pair<RowVectorXd,RowVectorXd> generate_training_example(istream& images, istream& labels){
		uint8_t buff[IMAGE_SIZE];
		images.seekg(example_num * IMAGE_SIZE, ios_base::cur);
		labels.seekg(example_num, ios_base::cur);
		//if(images.fail() || labels.fail()){cout << -example_num << endl; exit(0);}
		char label = labels.get();
		images.read( (char*)buff, IMAGE_SIZE);
		RowVectorXd image(IMAGE_SIZE + 1);
		image[0] = 1;
		for(int i = 0; i < IMAGE_SIZE; i++){
			image[i + 1] = buff[i];
		}
		RowVectorXd target = RowVectorXd::Constant(num_output_neurons + 1, 0);
		target[label + 1] = 1;
		target[0] = 1; //bias
		images.seekg(-example_num * IMAGE_SIZE - IMAGE_SIZE, ios_base::cur);
		labels.seekg(-example_num - 1, ios_base::cur);
		//cout << images.fail() << " " << labels.fail() << labels.tellg() << endl;
		//if(images.fail() || labels.fail()){cout << example_num << endl; exit(0);}

		return pair<RowVectorXd,RowVectorXd>(image,target);


	}


	//ELU Function Kevin Yan
	double ELU(double x) {
	if(x > 0)
		return x;
	else
		return elu_weight*(exp(x)-1);
  }
  //ELU derivative Function Kevin Yan
  double dELU(double x) {
    if(x > 0)
      return 1;
    else
      return ELU(x) + elu_weight;
	}

	RowVectorXd evaluate(const RowVectorXd& input_vector, const RowVectorXd& reference_vec){
		RowVectorXd temp = input_vector;
		int i = 0;
		for(MatrixXd l : layers){
			if(is_training) layer_inputs[i] = temp;
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
		//Mean Square loss
		RowVectorXd diff = reference_vec - input_vector;
		return diff.dot(diff); //feel free to change this to something faster
		//RowVectorXd log_in = input_vector.array().log().matrix();
		//return log_in.dot(reference_vec);
	}

	/*ArrayXd d_loss(const RowVectorXd& input_vector, const RowVectorXd& reference_vec){
		Array<double, 1,-1> out = (input_vector.array() * reference_vec.array()).transpose();
		return out.inverse();

	}*/


	//Learning



	void train(const string& image_file, const string& label_file){
		vector<MatrixXd> momenta; //hold the momentum vectors for each weight column
		for(MatrixXd l : layers){
			momenta.emplace_back(MatrixXd::Zero(l.rows(), l.cols()));
		}
		for(int i = 0; i < epochs; i++){
			is_training = true;
			ifstream images(image_file, ios::binary);
			ifstream labels(label_file, ios::binary);
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
			double prev_loss = 9999999;
			double loss_ex = 999999;

			vector<int> order(training_size);
			for(int n = 0; n < training_size; n++){
				order[n] = n;
			}
			random_shuffle(order.begin(), order.end());
			for(int k = 0; k < training_size; k++){
				//for(int g = 0; g  < BATCH_SIZE; g++){
				example_num = order[k];
				pair<RowVectorXd,RowVectorXd> example = generate_training_example(images, labels);
				RowVectorXd result = evaluate(example.first, example.second);
				prev_loss = loss_ex;
				loss_ex = loss(result, example.second);
				if(example_num % 10000 == 0){
					double max = 0;
					int argmax = 1;
					for(int z = 1; z < result.cols(); z++){
							if(result[z] > max){max = result[z]; argmax = z;}
					}
					cout << argmax - 1 << " | " << example.second << " | " << loss_ex <<  " | " << result <<  endl;
					if(loss_ex != loss_ex){
						cerr << "Network diverged!" << endl;
						exit(45);
					}
				}
				
				MatrixXd& final_layer = layers.back();
				RowVectorXd d_diff(final_layer.cols());

				RowVectorXd diff = example.second - result;
				//ArrayXd d_loss_arr = d_loss(result, example.second);
				for(int j = 0; j < final_layer.cols(); j++) d_diff[j] = dELU(result[j]);
				RowVectorXd sigma_v = (diff.array() * d_diff.array()).matrix(); //array allows for simple component-wise ops
				RowVectorXd sigma_v_next;
				for( int m = layers.size() - 1; m >= 0; m--){

					if(m > 0){
						sigma_v_next = RowVectorXd(layers[m - 1].cols());
						for(int j = 0; j<  sigma_v_next.cols(); j++){

							sigma_v_next[j] = layers[m].row(j).dot(sigma_v) * dELU(layer_inputs[m][j]);
						}
					}
					for(int j = 1; j < layers[m].cols();j++){	
						RowVectorXd temp =((learning_rate * sigma_v[j] * layer_inputs[m])	+ (momentum_constant * momenta[m].col(j).transpose()));
						//momenta[m].col(j) = temp;
						layers[m].col(j) += temp;
					}
					sigma_v = sigma_v_next;
				}
		}
		cout << endl;
		images.close();
		labels.close();
	}

	cout << "training complete!" << endl;
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
			for(int i = 1; i < num_hidden_neurons+1; i++) {
				hidden_out[0][i] = ELU(hidden_in[0][i]);
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
			for(int i = 1; i < num_output_neurons+1; i++) {
				output_out[i] = output_in[i];
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

		// for(MatrixXd m : layers) {
		// 	cout << m << endl;
		// }

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
			testing_labels.read( &label, sizeof(char));
			//run inputs through nn
			perceptron();
			//get nn prediction
			int prediction = 1;
			for(int i = 2; i < num_output_neurons + 1; i++) {
				// cout << output_out[i] << endl;
				if(output_out[i] > output_out[prediction]) {
					prediction = i;
				}
			}
			//cout << prediction << "==" << label + 1 << endl;
			if(prediction == label+1) {
				correctCount++;
			}
		}
		cout << "Correct Count: " << correctCount << endl;
		cout << "Total Count: " << num_images << endl;
	}

	//save weights Carl?
};

int main(int argc, char** argv){

	unsigned int seed = (unsigned int) time(0); //use seed 1576220813 for a good seed if you get *really* unlucky
	cout << "Seed: " << seed << endl;
	srand(seed);
	double learning_rate = (argc < 2)? .0004 : stod(string(argv[1]));
	int num_layers = (argc < 3)? 4: stoi(argv[2]);
	int epochs = (argc < 4)? 4: stoi(argv[3]);
	int hidden_layer_size = (argc < 5)? 16: stoi(argv[4]);
	double momentum = (argc < 6)? .9:stod(argv[5]);
	double elu_weight = (argc < 7)? .2 : stod(argv[6]);
	NeuralNetwork net(learning_rate, num_layers, epochs, hidden_layer_size, momentum, elu_weight);
	net.train(TRAINING_IMAGES_FILENAME, TRAINING_LABELS_FILENAME);
	net.testing(TESTING_IMAGES_FILENAME, TESTING_LABELS_FILENAME);

}

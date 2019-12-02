#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <Eigen/Dense>
#include <cmath>

using namespace std;

//num input neurons
int num_input_neurons;
//num hidden neurons
int num_hidden_neurons;
//num output neurons
int num_output_neurons;
//epochs?
//learning rate
double learning_rate;


using namespace Eigen;


const string training_images_filename = "train-images-idx3-ubyte";
const string training_labels_filename = "train-labels-idx1-ubyte";
const string testing_images_filename = "t10k-images-idx3-ubyte";
const string testing_labels_filenames = "t10k-labels-idx1-ubyte";
const int IMAGE_ROWS = 28
const int IMAGE_COLS = 28
const int IMAGE_SIZE =  IMAGE_ROWS * IMAGE_COLS; //28*28, the number of pixels in a MNIST image


//WHYYY are MNIST numbers in big endian
int read_num(char* buff, size_t size){
	int out = 5;
	char* test_p = &test;
	if(*test_p){ //system is little endian, make sure to reverse the number
		for(size_t i = size - 1; i <= 0; i+--){
			*test_p = buff + i;
			*test_p++;
		}
	}
	else{
		for(size_t i = 0; i < size; i++){
			*test_p = buff + i;
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

	//num hidden layers

	int num_input_neurons;
	//num hidden neurons
	int num_hidden_neurons;
	//num output neurons
	int num_output_neurons;
	int epochs;
	//learning rate
	double learning_rate;
	vector<MatrixXd> layers;
	mutex loss_sum_lock, producer, consumer, generate;
	condition_variable full, empty;
	vector<pair<VectorXd>> vecs_to_calc; //put input and expected output vectors into this, and take them out to process
	unsigned int pro_buffer_index, con_buffer_index, buffer_size, con_training_total, pro_training_total, buffer_taken;
	unsigned int num_producers, num_consumers;

	NeuralNetwork(double learning_rate, int num_layers, int epochs, int hidden_layer_size, int buffer_size){
		num_input_neurons = IMAGE_SIZE;
		num_output_neurons = 10;
		this->learning_rate = learning_rate;
		this->buffer_size = buffer_size
		pro_buffer_index = con_buffer_index = con_training_total = buffer_taken = 0;
		vecs_to_calc = vector(buffer_size);
		/* We use an extra hidden "neuron" to perpetuate the bias term, the
		   first element of each set of inputsoutput
		*/
		layers.emplace(layers.end(), MatrixXd::Random(num_input_neurons + 1, hidden_layer_size + 1));
		for(int i = 0; i  < num_layers - 2; i++){
			layers.emplace(layers.end(), MatrixXd::Random(hidden_layer_size + 1, hidden_layer_size + 1));
		}
		layers.emplace(layers.end(), MatrixXd::Random(hidden_layer_size + 1, num_output_neurons));
		this->epochs = epochs;



		//TODO: finish this

	}

	//Image

	pair<VectorXd> generate_training_example(istream& images, istream& labels){
		//TODO: finish this
		char buff[IMAGE_SIZE];
		auto l = unique_lock(generate);
		images.read(buff, IMAGE_SIZE);
		l.unlock();
		VectorXd target();

	}

	void producer_thread(istream& images, istream& labels, unsigned int training_size){
		while(images && pro_training_total < training_size){
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
	//This returns the loss for a single example, preconverted into a vector
	//Always pass Eigen matrices & vectors by reference!
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

	void train(const& string image_file, const string& label_file){
		for(int i = 0; i < epochs; i++){
			double loss = 0;
			ifstream images(image_file);
			ifstream labels(label_file);
			char magic_buff[4];
			images.read(magic, 4);
			int magic = read_int(magic_buff, 4);

			if(magic != 2051){
				cerr << "Bad image data! " << magic << endl;
				exit(magic);
			}
			labels.read(magic_buff, 4);
			magic = read_int(magic_buff, 4);
			if(magic != 2049){
				cerr << "Bad label data! " << magic << endl;
				exit(magic);
			}

			//TODO: read out dimensions and training size


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


			//backpropagate

			f.close();
		}


	}
	//method to test (Kevin Yan)
	void testing(vector<MatrixXd> nn, string testing_images_filename, string testinglabels_filename) {
		ifstream testing_images;
		ifstream testing_labels;
		int image_magic_num;
		int label_magic_num;
		int num_images;
		int num_labels;
		int num_rows;
		int num_cols;
		char buffer;
		int image_matrix[IMAGE_ROWS][IMAGE_COLS];
		//read binary image and label files
		testing_images.open(testing_images_filename,ios::binary);
		testing_labels.open(testing_labels_filename,ios::binary);
		//read in image headers
		testing_images.read((char*)&image_magic_num,sizeof(image_magic_num));
		testing_images.read((char*)&num_images,sizeof(num_images));
		testing_images.read((char*)&num_rows,sizeof(num_rows));
		testing_images.read((char*)&num_cols,sizeof(num_cols));

		//read in label headers
		testing_labels.read((char*)&label_magic_num,sizeof(label_magic_num));
		testing_labels.read((char*)&num_labels,sizeof(num_labels));

		//check magic numbers
		if(image_magic_num != 2051) {
			cerr << "Bad Image Data! " << image_magic_num << endl;
			exit(image_magic_num);
		}
		if(label_magic_num != 2049){
			cerr << "Bad label data! " << magic << endl;
			exit(label_magic_num);
		}

		//read image data
		//grayscale
		for(int i = 0; i < IMAGE_ROWS; i++) {
			for(int j = 0; j < IMAGE_COLS; j++) {
				testing_images.read(&buffer, sizeof(char));
				buffer == 0 ? image_matrix[i][j] = 0 : image_matrix[i][j] = 1;
			}
		}

		int predction = 0;
		for(int i = 1; i < num_output_neurons; i++) {
			if(SUBWITHOUTPUT[i] > SUBWITHOUTPUT[prediction]) {
				prediction = i;
			}
		}



	}






	//Reading input Carl

	//save weights
};
//main

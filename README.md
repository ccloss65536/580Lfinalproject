# 580Lfinalproject
Final project repository for Fall 2019 Machine Learning
Use makefile to compile:
make

Run:
./training.exe <learning rate> <total layers> <epochs> <size of hidden layers> <momentum> <a(for elu)> <l2 penalty>
  
This trains and then tests the neural network using the MNIST dataset and the given parameters.

Although the code trains and converges seemingly normally, we are getting an accuracy rate of about 11% which is very close to random.
As a result, we are unable to get a proper result for the accuracy of the program, but are still able to get other performance measures
such as power consumption.

Contributions:
Carl Closs- NN Training and some of Testing
Kevin Yan- NN Testing Functionality
Brian Grant- Testing and Performance Measures

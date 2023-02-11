#include <iostream>
#include <Dataset/dataset.h>
#include <Preprocessing/normalization.h>
#include <Regression/linear.h>
#include <Regression/logistic.h>
#include <Loss/mean_squared_error.h>
#include <Loss/binary_cross_entropy.h>
#include <Optimizers/bgd.h>
#include <Optimizers/sgd.h>

#include <Eigen3/Eigen/Dense>
using Eigen::MatrixXd;

#include <examples.h>

// Display functions

void display_data(MatrixXd x_train, MatrixXd y_train, 
	MatrixXd weights, double bias, double learning_rate){
	std::cout << "\nx_train = " << std::endl << x_train << "\n" << std::endl; 
	std::cout << "y_train = " << std::endl << y_train << "\n" << std::endl; 
	std::cout << "weights = " "[" << weights.transpose() << "]" << std::endl; 
	std::cout << "bias = " << bias << std::endl;
	std::cout << "learning rate = " << learning_rate << "\n" << std::endl; 
}

void display_results(MatrixXd y_predict, MatrixXd y_train){
	std::cout << "\nTraining Complete.\n" << std::endl <<
	"prediction: [" << y_predict.transpose() << "]" << std::endl <<
	"training: [" << y_train.transpose() << "]" << std::endl;
}

int main()
{
	std::cout << "MLcpp: Machine Learning library built with C++" << std::endl;

	// example_linear_regression_bgd();
	// example_linear_regression_sgd();
	// example_logistic_regression();
}
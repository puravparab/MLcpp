#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Regression/linear.h>

using Eigen::MatrixXd;

int main()
{
	std::cout << "Machine Learning library built with C++" << std::endl;

	// Linear Regression
	MatrixXd x_train(2,1);
	x_train(0,0) = 1.0;
	x_train(1,0) = 2.0;

	MatrixXd y_train(2,1);
	y_train(0,0) = 300;
	y_train(1,0) = 500;

	MatrixXd weights(1,1);
	weights(0,0) = 200;

	double bias = 100;

	std::cout << "x_train = " << std::endl << x_train << "\n" << std::endl; 
	std::cout << "weights = " << std::endl << weights << "\n" << std::endl; 
	std::cout << "bias = " << bias << "\n" << std::endl; 

	std::cout << "Creating linear model ... " << std::endl; 

	Linear linear(x_train, weights, bias);
	MatrixXd y_predict = linear.train();

	for (int i = 0; i < y_train.cols(); i++){
		std::cout << i + 1 << "th step" << std::endl;
		if (y_train(i,0) != y_predict(i,0)){
			std::cout << "Training Falied" << std::endl;
		}
	}
	std::cout << "Training Complete" << std::endl;
}
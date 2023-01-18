#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Regression/linear.h>
#include <Loss/mean_squared_error.h>

using Eigen::MatrixXd;

int main()
{
	std::cout << "Machine Learning library built with C++" << std::endl;

	// Linear Regression
	MatrixXd x_train(4,1);
	x_train << 2.1040, 1.4160, 1.5340, 0.8520;

	MatrixXd y_train(4,1);
	y_train << 4.000, 2.320, 3.150, 1.780;

	MatrixXd weights(1,1);
	weights(0,0) = 5.0;

	double bias = 1.0;

	double learning_rate = 0.00001;

	std::cout << "x_train = " << std::endl << x_train << "\n" << std::endl; 
	std::cout << "y_train = " << std::endl << y_train << "\n" << std::endl; 
	std::cout << "weights = " "[" << weights.transpose() << "]" << "\n" << std::endl; 
	std::cout << "bias = " << bias << "\n" << std::endl; 

	std::cout << "Creating linear model ... " << std::endl; 

	Linear linear(x_train, y_train, weights, bias);
	MatrixXd y_predict = linear.train(learning_rate);
	
	std::cout << "\nTraining Complete.\n" << std::endl <<
	"prediction: [" << y_predict.transpose() << "]" << std::endl <<
	"training: [" << y_train.transpose() << "]" << std::endl;

	MatrixXd x_i(1,1);
	x_i(0,0) = 2;
	std::cout << "Prediction for x = 1.2: " << linear.predict(x_i) << std::endl;
}
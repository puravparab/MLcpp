#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Regression/linear.h>
#include <Loss/mean_squared_error.h>

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
	std::cout << "weights = " "[" << weights.transpose() << "]" << "\n" << std::endl; 
	std::cout << "bias = " << bias << "\n" << std::endl; 

	std::cout << "Creating linear model ... " << std::endl; 

	Linear linear(x_train, weights, bias);
	MatrixXd y_predict = linear.train();
	
	for (int i = 0; i < y_train.rows(); i++){
		std::cout << "Training example #" << i + 1 << std::endl;
	}
	std::cout << "Training Complete." << "\n\n" << 
	"prediction: [" << y_predict.transpose() << "]" << std::endl <<
	"training: [" << y_train.transpose() << "]" << std::endl;

	// Loss:
	MeanSquaredError Loss(y_predict, y_train);
	double error = Loss.get_error();
	std::cout << error << std::endl;

	MatrixXd x_i(1,1);
	x_i(0,0) = 1.2;
	std::cout << "Prediction for x = 1.2: " << linear.predict(x_i) << std::endl;
}
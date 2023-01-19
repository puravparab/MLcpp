#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Regression/linear.h>
#include <Loss/mean_squared_error.h>
#include <Preprocessing/normalization.h>

using Eigen::MatrixXd;

int main()
{
	std::cout << "Machine Learning library built with C++" << std::endl;

	// // Linear Regression:
	// MatrixXd x_train(4,1);
	// x_train << 2.1040, 1.4160, 1.5340, 0.8520;
	// MatrixXd y_train(4,1);
	// y_train << 4.000, 2.320, 3.150, 1.780;
	// MatrixXd weights(1,1);
	// weights << 5.0;
	// double bias = 1.0;
	// double learning_rate = 0.00003;

	// Multiple Regression:
	MatrixXd x_train{
		{2104, 5, 1, 45},
		{1415, 3, 2, 40},
		{852, 2, 1, 35}
	};
	MatrixXd y_train{
		{460},
		{232},
		{178}
	};
	MatrixXd weights{
		{0.39},
		{18.75},
		{-53.36},
		{-26.421}
	};
	double bias = 785.18;
	double learning_rate = 5.0e-5;

	// Normalize Input
	Normalization normalized(x_train);
	x_train = normalized.get_x_train();
	
	std::cout << "\nx_train = " << std::endl << x_train << "\n" << std::endl; 
	std::cout << "y_train = " << std::endl << y_train << "\n" << std::endl; 
	std::cout << "weights = " "[" << weights.transpose() << "]" << std::endl; 
	std::cout << "bias = " << bias << std::endl;
	std::cout << "learning rate = " << learning_rate << "\n" << std::endl; 

	std::cout << "Creating linear model ... " << std::endl; 

	Linear linear(x_train, y_train, weights, bias);
	MatrixXd y_predict = linear.train(learning_rate);
	
	std::cout << "\nTraining Complete.\n" << std::endl <<
	"prediction: [" << y_predict.transpose() << "]" << std::endl <<
	"training: [" << y_train.transpose() << "]" << std::endl;

	//  Prediction:

	// MatrixXd x_i{
	// 	{2},
	// 	{1}
	// };
	MatrixXd x_i{
		{2000, 4, 2, 30},
		{1000, 2, 1, 35}
	};

	x_i = normalized.process(x_i);
	std::cout << "Prediction for x:\n" << linear.predict(x_i) << std::endl;
}
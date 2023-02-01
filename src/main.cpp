#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Dataset/dataset.h>
#include <Regression/linear.h>
#include <Regression/logistic.h>
#include <Loss/mean_squared_error.h>
#include <Loss/binary_cross_entropy.h>
#include <Preprocessing/normalization.h>

using Eigen::MatrixXd;

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

	// std::string url = ".\\dataset\\binary_test1.csv";
	// Dataset data(url);
	// MatrixXd x_train = data.get_x_train();
	// MatrixXd y_train = data.get_y_train();

	// std::cout << "training examples: " << x_train.rows() << std::endl;
	// MatrixXd weights{
	// 	{5},{5}
	// };
	// double bias = -14;
	// double learning_rate = 5e-4;

	// Logistic logistic(x_train, y_train, weights, bias);
	// logistic.train(learning_rate, "bgd");

	// MatrixXd y_predict = logistic.predict();
	// std::cout << "Predictions: \n" << y_predict << std::endl;

	// Multiple Regression:
	std::string url = ".\\dataset\\real_estate.csv";
	Dataset dataset(url, 80);
	MatrixXd train = dataset.get_train();
	MatrixXd test = dataset.get_test();

	// Get training data
	MatrixXd x_train = train.block(0, 0, train.rows(), train.cols() - 1);
	MatrixXd y_train = train.col(train.cols() - 1);
	y_train = y_train / 1000000; // Scale training target

	// Get test data
	MatrixXd x_test = test.block(0, 0, test.rows(), test.cols()-1);
	MatrixXd y_test = test.col(test.cols()-1);
	y_test = y_test / 1000000; // Scale test target

	std::cout << "training examples: " << x_train.rows() << " test examples: " << x_test.rows() << std::endl;
	// Normalize Input
	Normalization normalized(x_train);
	x_train = normalized.get_x_train();

	MatrixXd weights{
		{100},{100},{100},{100}
	};
	double bias = 0;
	double learning_rate = 1e-3;
	
	Linear linear(x_train, y_train, weights, bias);
	MatrixXd y_predict = linear.train(learning_rate, "bgd");

	x_test = normalized.process(x_test);
	std::cout << linear.evaluate(x_test, y_test) << std::endl;

	// x1: bedrooms = 5
	// x2: bathrooms = 3
	// x3: size of home (sqft) = 2400
	// x4: size of lot (sqft) = 3000
	// y = price of home (dollars)
	MatrixXd x{
		{2, 4, 2400, 3000}
	};
	x = normalized.process(x);
	std::cout << "Prediction: \n" << linear.predict(x) * 100000 << " dollars" << std::endl;
}
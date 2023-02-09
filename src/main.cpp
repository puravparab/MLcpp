#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Dataset/dataset.h>
#include <Regression/linear.h>
#include <Regression/logistic.h>
#include <Loss/mean_squared_error.h>
#include <Loss/binary_cross_entropy.h>
#include <Preprocessing/normalization.h>
#include <Optimizers/bgd.h>

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

	// Logistic Regression:
	std::string url = ".\\dataset\\binary_test1.csv";
	Dataset data(url);
	MatrixXd train = data.get_train();
	MatrixXd x_train = train.block(0, 0, train.rows(), train.cols() - 1);
	MatrixXd y_train = train.col(train.cols() - 1);

	std::cout << "training examples: " << x_train.rows() << std::endl;
	MatrixXd weights{
		{0},{0}
	};
	double bias = 0;
	double learning_rate = 0.3;
	double epsilon = 1e-6;
	double iterations = 5000;

	// Train logistic regression model
	Logistic logistic(x_train, y_train, weights, bias);
	MatrixXd y_predict = logistic.train(learning_rate, "bgd", epsilon, iterations, 200);
	// MatrixXd x{
	// 	{2.5, 3},
	// 	{0.5, 0.5}
	// };
	std::cout << "\nPredictions: \n" << y_predict << std::endl;

	MatrixXd x{
		{2.5, 3},
		{0.5, 0.5}
	};
	std::cout << "\nPredictions:\n" << logistic.predict(x) << std::endl;
	std::cout << "\nTest:\n" << logistic.evaluate(x_train, y_train) << std::endl;


	// // Multiple Regression:
	// std::string url = ".\\dataset\\real_estate.csv";
	// Dataset dataset(url);
	// MatrixXd train = dataset.get_train();
	// MatrixXd test = dataset.get_test();

	// int scale = 1000000;
	// // Get training data
	// MatrixXd x_train = train.block(0, 0, train.rows(), train.cols() - 1);
	// MatrixXd y_train = train.col(train.cols() - 1);
	// y_train = y_train / scale; // Scale training target

	// // Get test data
	// MatrixXd x_test = test.block(0, 0, test.rows(), test.cols()-1);
	// MatrixXd y_test = test.col(test.cols()-1);
	// y_test = y_test / scale; // Scale test target

	// std::cout << "training examples: " << x_train.rows() << " test examples: " << x_test.rows() << std::endl;
	// // Normalize Input
	// Normalization normalized(x_train);
	// x_train = normalized.get_x_train();

	// MatrixXd weights{
	// 	{0},{0},{0},{0}
	// };

	// double bias = 0;
	// double learning_rate = 2e-2;
	// double epsilon = 1e-5;
	// double iterations = 100000;
	// Linear linear(x_train, y_train, weights, bias);
	// MatrixXd y_predict = linear.train(learning_rate, "bgd", epsilon, iterations, 1000);

	// x_test = normalized.process(x_test);
	// std::cout << "Test error: " << linear.evaluate(x_test, y_test) << std::endl;

	// // x1: bedrooms
	// // x2: bathrooms
	// // x3: size of home (sqft)
	// // x4: size of lot (sqft)
	// // y = price of home (dollars)
	// MatrixXd x{
	// 	{2, 4, 2400, 3000},
	// 	{4, 6, 2800, 3200}
	// };
	// x = normalized.process(x);
	// std::cout << "Predictions: \n" << linear.predict(x) * scale << " dollars" << std::endl;
}
#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Dataset/dataset.h>
#include <Regression/linear.h>
#include <Loss/mean_squared_error.h>
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
	std::cout << "Machine Learning library built with C++" << std::endl;

	// Univariate Linear Regression:
	std::string url = ".\\dataset\\test1.csv";
	Dataset data1(url);
	MatrixXd x_train = data1.get_x_train();
	MatrixXd y_train = data1.get_y_train();

	MatrixXd weights(1,1);
	weights << 560.0;
	double bias = 10.0;
	double learning_rate = 2e-5;

	// Normalize Input
	Normalization normalizedLR(x_train);
	x_train = normalizedLR.get_x_train();

	display_data(x_train, y_train, weights, bias, learning_rate);

	std::cout << "Creating Linear model ...."<< std::endl;
	Linear linear1(x_train, y_train, weights, bias, "sgd");
	MatrixXd y_predict = linear1.train(learning_rate);

	display_results(y_predict, y_train);

	// // Multiple Regression:
	// url = ".\\dataset\\test2.csv";
	// Dataset data2(url);
	// x_train = data2.get_x_train();
	// y_train = data2.get_y_train();

	// MatrixXd weights2{
	// 	{0.39},
	// 	{18.75},
	// 	{-53.36},
	// 	{-26.421}
	// };
	// bias = 785.18;
	// learning_rate = 5.0e-5;

	// // Normalize Input
	// Normalization normalizedMR(x_train);
	// x_train = normalizedMR.get_x_train();
	
	// display_data(x_train, y_train, weights2, bias, learning_rate);

	// std::cout << "Creating Linear model ...."<< std::endl;
	// Linear linear2(x_train, y_train, weights2, bias);
	// y_predict = linear2.train(learning_rate);

	// display_results(y_predict, y_train);

	//  Predictions:
	MatrixXd x1{
		{2000},
		{1000}
	};
	// MatrixXd x2{
	// 	{2000, 4, 2, 30},
	// 	{1000, 2, 1, 35}
	// };

	x1 = normalizedLR.process(x1);
	std::cout << "Prediction for x (Univariate):\n" << linear1.predict(x1) << std::endl;

	// x2 = normalizedMR.process(x2);
	// std::cout << "Prediction for x (Muliple LR):\n" << linear2.predict(x2) << std::endl;

	// std::string url = ".\\dataset\\real_estate_data.csv";
	// Dataset data(url);
	// MatrixXd x_train = data.get_x_train();
	// MatrixXd y_train = data.get_y_train();

	// // Normalize Input
	// Normalization normalized(x_train);
	// x_train = normalized.get_x_train();

	// MatrixXd weights{
	// 	{500},{300},{300}
	// };
	// double bias = 100;
	// double learning_rate = 0.01;

	// Linear linear(x_train, y_train, weights, bias);
	// MatrixXd y_predict = linear.train(learning_rate);

	// MatrixXd x{
	// 	{5,3,3000}
	// };
	// x = normalized.process(x);
	// std::cout << "Prediction for x:\n" << linear.predict(x) << std::endl;
}
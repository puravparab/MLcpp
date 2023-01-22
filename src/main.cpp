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
	std::cout << "MLcpp: Machine Learning library built with C++" << std::endl;

	std::string url = ".\\dataset\\real_estate.csv";
	Dataset data(url);
	MatrixXd x_train = data.get_x_train();
	MatrixXd y_train = data.get_y_train();

	// Normalize Input
	Normalization normalized(x_train);
	x_train = normalized.get_x_train();

	MatrixXd weights{
		{0},{0},{0},{0}
	};
	double bias = 0;
	double learning_rate = 0.01;

	Linear linear(x_train, y_train, weights, bias, "sgd");
	MatrixXd y_predict = linear.train(learning_rate);

	// x1: bedrooms = 5
	// x2: bathrooms = 3
	// x3: size of home (sqft) = 2400
	// x4: size of lot (sqft) = 3000
	// y = price of home (dollars)
	MatrixXd x{
		{5, 3, 2400, 3000}
	};
	x = normalized.process(x);
	std::cout << "Prediction: \n" << linear.predict(x) << " dollars" << std::endl;
}
// Logctic Regression
// logistic.h

#ifndef LOGISTIC_H
#define LOGISTIC_H

#include <Eigen3/Eigen/Dense>
using Eigen::MatrixXd;

class Logistic{
	MatrixXd x; // Features (n examples, m features)
	MatrixXd y; // Target (n examples, 1 column)
	MatrixXd w; // Weights (n rows, 1 column)
	double b; // Bias

	MatrixXd x_test;
	MatrixXd y_test;

	public:
	Logistic(MatrixXd x, MatrixXd y, MatrixXd weights, double bias)
		: x(x), y(y), w(weights), b(bias), x_test(x), y_test(y)
		{}
	
	Logistic(MatrixXd x, MatrixXd y, MatrixXd weights, double bias, MatrixXd x_test, MatrixXd y_test)
		: x(x), y(y), w(weights), b(bias), x_test(x_test), y_test(y_test)
		{}

	MatrixXd train(double learning_rate, std::string gradient_descent);
	MatrixXd predict();
	MatrixXd predict(MatrixXd x_i);
};

#endif // LOGISTIC_H
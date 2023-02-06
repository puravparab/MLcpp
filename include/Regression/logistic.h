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

	public:
	Logistic(MatrixXd x, MatrixXd y, MatrixXd weights, double bias)
		: x(x), y(y), w(weights), b(bias)
		{}

	MatrixXd train(double learning_rate, std::string gradient_descent, double epsilon, int iteration, int iteration_skip);
	MatrixXd predict();
	MatrixXd predict(MatrixXd x_i);
};

#endif // LOGISTIC_H
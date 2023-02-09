// Linear Regression
// linear.h
#ifndef LINEAR_H
#define LINEAR_H

#include <Eigen3/Eigen/Dense>
using Eigen::MatrixXd;

class Linear{
	MatrixXd x; // Features (n examples, m features)
	MatrixXd y; // Target (n examples, 1 column)
	MatrixXd w; // Weights (n rows, 1 column)
	double b; // Bias

	public:
	Linear(MatrixXd x, MatrixXd y, MatrixXd weights, double bias)
		: x(x), y(y), w(weights), b(bias)
		{}
	
	MatrixXd train(MatrixXd x_test, MatrixXd y_test, double learning_rate, std::string gradient_descent, double epsilon, int iteration, int iteration_skip);
	MatrixXd predict();
	MatrixXd predict(MatrixXd x_i);
	double evaluate(MatrixXd x_test, MatrixXd y_test);
};

#endif /* LINEAR_H */
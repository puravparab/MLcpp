// linear.h
#ifndef LINEAR_H
#define LINEAR_H

#include <Eigen3/Eigen/Dense>
using Eigen::MatrixXd;

class Linear{
	MatrixXd x; // Features (n examples, m features)
	MatrixXd y; // Target (n examples, 1 column)
	MatrixXd w; // Weights (n rows, 1 column)

	MatrixXd x_test;
	MatrixXd y_test;
	double b; // Bias
	public:
		Linear(MatrixXd x, MatrixXd y, MatrixXd weights, double bias)
			: x(x), y(y), w(weights), b(bias), x_test(x), y_test(y)
			{}
		
		Linear(MatrixXd x, MatrixXd y, MatrixXd weights, double bias, MatrixXd x_test, MatrixXd y_test)
			: x(x), y(y), w(weights), b(bias), x_test(x_test), y_test(y_test)
			{}
		
		MatrixXd train(double learning_rate, std::string gradient_descent);
		MatrixXd predict();
		MatrixXd predict(MatrixXd x_i);
};

#endif /* LINEAR_H */
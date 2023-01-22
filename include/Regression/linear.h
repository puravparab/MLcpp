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
		
		MatrixXd train(double learning_rate, std::string gradient_descent);
		MatrixXd predict();
		MatrixXd predict(MatrixXd x_i);
};

#endif /* LINEAR_H */
// linear.h
#ifndef LINEAR_H
#define LINEAR_H

#include <Eigen3/Eigen/Dense>
using Eigen::MatrixXd;

class Linear{
	MatrixXd x; // Features (n examples, m features)
	MatrixXd w; // Weights (n rows, 1 column)
	double b; // Bias

	public:
		Linear(MatrixXd x, MatrixXd weights, double bias)
			: x(x), w(weights), b(bias)
			{}
		
		MatrixXd train();
		double predict(MatrixXd x_i);
};

#endif /* LINEAR_H */
// linear.h
#ifndef LINEAR_H
#define LINEAR_H

#include <Eigen3/Eigen/Dense>
using Eigen::MatrixXd;

class Linear{
	MatrixXd x; // Features
	MatrixXd w; // Weights
	double b; // Bias

	public:
		Linear(MatrixXd x, MatrixXd weights, double bias)
			: x(x), w(weights), b(bias)
			{}
		
		MatrixXd train();
		double predict(double x_i);
};

#endif /* LINEAR_H */
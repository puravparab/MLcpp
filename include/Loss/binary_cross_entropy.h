// Binary Cross Entropy Loss
// binary_cross_entropy.h

#ifndef BINARY_CROSS_ENTROPY_H
#define BINARY_CROSS_ENTROPY_H

#include <Eigen3/Eigen/Dense>
using Eigen::MatrixXd;

class BinaryCrossEntropy{
	MatrixXd y_predict; // Model Predictions
	MatrixXd y_train; // Training output
	MatrixXd x_train; // Training input
	public:
		BinaryCrossEntropy(MatrixXd y_predict, MatrixXd y_train, MatrixXd x_train)
			: y_predict(y_predict), y_train(y_train), x_train(x_train)
			{}
		
		double get_error();
		MatrixXd get_derivative_w();
		double get_derivative_b();
};

#endif /* MEAN_SQUARED_ERROR */
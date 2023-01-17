// mean_squared_error.h
#ifndef MEAN_SQUARED_ERROR_H
#define MEAN_SQUARED_ERROR_H

#include <Eigen3/Eigen/Dense>
using Eigen::MatrixXd;

class MeanSquaredError{
	MatrixXd y_predict; // Model Predictions
	MatrixXd y_train; // Training output
	MatrixXd x_train; // Training input
	public:
		MeanSquaredError(MatrixXd y_predict, MatrixXd y_train, MatrixXd x_train)
			: y_predict(y_predict), y_train(y_train), x_train(x_train)
			{}
		
		double get_error();
		double get_derivative_w();
		double get_derivative_b();
};

#endif /* MEAN_SQUARED_ERROR */
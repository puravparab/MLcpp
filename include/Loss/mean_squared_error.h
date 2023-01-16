// mean_squared_error.h
#ifndef MEAN_SQUARED_ERROR_H
#define MEAN_SQUARED_ERROR_H

#include <Eigen3/Eigen/Dense>
using Eigen::MatrixXd;

class MeanSquaredError{
	MatrixXd y_predict; // Model Predictions
	MatrixXd y_train; // Training output
	public:
		MeanSquaredError(MatrixXd y_predict, MatrixXd y_train)
			: y_predict(y_predict), y_train(y_train)
			{}
		
		double get_error();
};

#endif /* MEAN_SQUARED_ERROR */
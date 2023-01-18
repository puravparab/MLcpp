#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Loss/mean_squared_error.h>

/* Mean Squared Error:

	J(w,b) = (1/2n) x sum[1,n](y_predict[i] - y_train[i])^2

	where:
	 n = number of training examples
	 i = ith training example
*/

double MeanSquaredError::get_error(){
	MatrixXd y_diff = y_predict - y_train;
	for (int i = 0; i < y_diff.rows(); i++){
		y_diff(i,0) = pow(y_diff(i,0), 2);
	}
	MatrixXd error = (y_diff.colwise().sum()) / (2.0 * y_predict.rows());
	return error(0,0);
}

// Does not work with multiple features
double MeanSquaredError::get_derivative_w(){
	MatrixXd y_diff = y_predict - y_train;
	for (int i = 0; i < y_diff.rows(); i++){
		y_diff(i,0) = y_diff(i,0) * x_train(i,0);
	}
	MatrixXd error = (y_diff.colwise().sum()) / (1.0 * y_predict.rows());
	return error(0,0);
}

// Does not work with multiple features
double MeanSquaredError::get_derivative_b(){
	MatrixXd y_diff = y_predict - y_train;
	MatrixXd error = (y_diff.colwise().sum()) / (1.0 * y_predict.rows());
	return error(0,0);
}
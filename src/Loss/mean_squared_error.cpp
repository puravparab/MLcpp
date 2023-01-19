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
	Eigen::ArrayXXd y_diff = y_predict - y_train;
	y_diff = y_diff * y_diff;
	MatrixXd error = (y_diff.colwise().sum()) / (2.0 * y_predict.rows());
	return error(0,0);
}

MatrixXd MeanSquaredError::get_derivative_w(){
	MatrixXd y_diff = y_predict - y_train;
	MatrixXd derivative_cost = y_diff.transpose() * x_train;
	return derivative_cost.transpose();
}

double MeanSquaredError::get_derivative_b(){
	MatrixXd y_diff = y_predict - y_train;
	MatrixXd error = (y_diff.colwise().sum());
	return error(0,0);
}
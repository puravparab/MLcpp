#include <iostream>
#include <math.h>
#include <Eigen3/Eigen/Dense>
#include <Loss/binary_cross_entropy.h>

/* Binary Cross Entropy :

	J(w,b) = (1/n) x sum[1,n](
		-y_train[i] x log(y_predict[i]) - (1 - y_train[i]) x log(1 - y_predict[i])
	)

	where:
	 n = number of training examples
	 i = ith training example
*/

double BinaryCrossEntropy::get_error(){
	// Eigen::ArrayXXd y_diff = y_predict - y_train;
	// y_diff = y_diff * y_diff;
	// MatrixXd error = (y_diff.colwise().sum()) / (2.0 * y_predict.rows());
	// return error(0,0);

	MatrixXd Ones = Eigen::MatrixXd::Ones(y_train.rows(), 1);
	MatrixXd logOne = y_predict.array().log();
	MatrixXd logTwo = (Ones - y_predict).array().log();
	MatrixXd L = -1 * (y_train.cwiseProduct(logOne) + ((Ones - y_train).cwiseProduct(logTwo)));

	MatrixXd error = (L.colwise().sum()) / (y_predict.rows());
	return error(0,0);
}

MatrixXd BinaryCrossEntropy::get_derivative_w(){
	MatrixXd Ones = Eigen::MatrixXd::Ones(y_train.rows(), 1);
	MatrixXd logOne = y_predict.array().log();
	MatrixXd logTwo = (Ones - y_predict).array().log();
	MatrixXd L = -1 * (y_train.cwiseProduct(logOne) + ((Ones - y_train).cwiseProduct(logTwo)));
	MatrixXd derivative_cost = L.transpose() * x_train;
	return derivative_cost.transpose();
}

double BinaryCrossEntropy::get_derivative_b(){
	// MatrixXd y_diff = y_predict - y_train;
	// MatrixXd error = (y_diff.colwise().sum());
	// return error(0,0);

	MatrixXd Ones = Eigen::MatrixXd::Ones(y_train.rows(), 1);
	MatrixXd logOne = y_predict.array().log();
	MatrixXd logTwo = (Ones - y_predict).array().log();
	MatrixXd L = -1 * (y_train.cwiseProduct(logOne) + ((Ones - y_train).cwiseProduct(logTwo)));
	MatrixXd derivative_cost = (L.colwise().sum());
	return derivative_cost(0,0);
}
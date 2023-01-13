#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Regression/linear.h>

using Eigen::MatrixXd;

MatrixXd Linear::train(){
	MatrixXd y_predict = MatrixXd::Constant(x.rows(), 1, 0);

	// Iterate through training examples:
	for (int i = 0; i < x.rows(); i++){
		double value = 0;
		// Iterate through features
		for (int j = 0; j < x.cols(); j++){
			value += x(i,j) * w(j,0);
		}
		y_predict(i, 0) = value + b;
	}
	return y_predict;
}

double Linear::predict(MatrixXd x_i){
	double y_predict = 0;
	for (int i = 0; i < w.rows(); i++){
		y_predict += w(i,0) * x_i(0,i);
	}
	y_predict += b;
	return y_predict;
}
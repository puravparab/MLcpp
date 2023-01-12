#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Regression/linear.h>

using Eigen::MatrixXd;

MatrixXd Linear::train(){
	MatrixXd y_predict = MatrixXd::Constant(x.rows(), 1, 0);

	for (int i = 0; i < x.rows() ; i++){
		y_predict(i, 0) = x(i, 0) * w(0, 0) + b;
	}
	return y_predict;
}

double Linear::predict(double x_i){
	double y_predict = w(0,0) * x_i + b;
	return y_predict;
}
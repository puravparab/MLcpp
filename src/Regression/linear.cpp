#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Regression/linear.h>
#include <Loss/mean_squared_error.h>
#include <Optimizers/bgd.h>
#include <Optimizers/sgd.h>
#include <chrono>

MatrixXd Linear::train(double learning_rate){
	MatrixXd y_predict = predict();
	BGD bgd(w, b, y_predict, y, x, learning_rate);
	
	MatrixXd* new_weights = new MatrixXd(w.rows(),w.cols());
	double* new_bias = new double;

	// Run optimizer
	bgd.optimize(new_weights, new_bias);

	// Update weight and bias
	w = *new_weights;
	b = *new_bias;
	y_predict = predict();
	delete new_weights;
	delete new_bias;
	return y_predict;
}

MatrixXd Linear::predict(){
	MatrixXd y_predict;
	MatrixXd b_vec = MatrixXd::Constant(y.rows(), 1, b);
	y_predict = x * w + b_vec;
	return y_predict;
}

MatrixXd Linear::predict(MatrixXd x_i){
	MatrixXd y_predict;
	MatrixXd b_vec = MatrixXd::Constant(x_i.rows(), 1, b);
	y_predict = x_i * w + b_vec;
	return y_predict;
}
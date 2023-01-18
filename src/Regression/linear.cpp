#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Regression/linear.h>
#include <Loss/mean_squared_error.h>
#include <Optimizers/sgd.h>

MatrixXd Linear::train(double learning_rate){
	MatrixXd y_predict = predict();
	SGD sgd(w, b, y_predict, y, x, learning_rate);

	MatrixXd* new_weights = new MatrixXd(w.rows(),w.cols());
	double* new_bias = new double;

	// Run optimizer
	sgd.optimize(new_weights, new_bias);

	// Update weight and bias
	w = *new_weights;
	b = *new_bias;
	y_predict = predict();
	delete new_weights;
	delete new_bias;
	return y_predict;
}

MatrixXd Linear::predict(){
	MatrixXd y = MatrixXd::Constant(x.rows(), 1, 0);
	// Iterate through training examples:
	for (int i = 0; i < x.rows(); i++){
		double value = 0;
		y(i,0) = predict(x.row(i));
	}
	return y;
}

double Linear::predict(MatrixXd x_i){
	double y = 0;
	for (int i = 0; i < w.rows(); i++){
		y += w(i,0) * x_i(0,i);
	}
	y += b;
	return y;
}
#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Regression/linear.h>
#include <Loss/mean_squared_error.h>
#include <Optimizers/sgd.h>
#include <chrono>

MatrixXd Linear::train(double learning_rate){
	// auto initial = std::chrono::high_resolution_clock::now();

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

	// auto final = std::chrono::high_resolution_clock::now() - initial;
	// long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(final).count();
	// std::cout << "Time: " << microseconds / 1e+6 << std::endl;
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
	MatrixXd b_vec = MatrixXd::Constant(y.rows(), 1, b);
	y_predict = x_i * w + b_vec;
	return y_predict;
}
#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Loss/mean_squared_error.h>
#include <Optimizers/sgd.h>

double SGD::update_weights(){
	double w_new = w(0,0) - learning_rate * mse.get_derivative_w();
	return w_new;
}

double SGD::update_bias(){
	double b_new = b - learning_rate * mse.get_derivative_b();
	return b_new;
}
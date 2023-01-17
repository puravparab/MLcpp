#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Loss/mean_squared_error.h>
#include <Optimizers/sgd.h>

double SGD::update_weights(){
	double w_new = w(0,0) - 0.5 * mse.get_derivative_w();
	return w_new;
}

double SGD::update_bias(){
	double b_new = b - 0.5 * mse.get_derivative_b();
	return b_new;
}
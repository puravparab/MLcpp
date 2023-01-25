#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Regression/linear.h>
#include <Loss/mean_squared_error.h>
#include <Optimizers/bgd.h>
#include <Optimizers/sgd.h>

// Train the linear model:
// Params:
// 	1. learning rate (double)
//	2. gradient_descent (string): "sgd" - stochastic, "bgd"- batch
// 
// Returns N x 1 matrix of predictions
MatrixXd Linear::train(double learning_rate, std::string gradient_descent){
	MatrixXd y_predict = predict();
	if (gradient_descent == "bgd"){
		BGD gd(w, b, y_predict, y, x, y_test, x_test, learning_rate, "mse");
		gd.optimize();
		w = gd.get_weight();
		b = gd.get_bias();
	}
	else if (gradient_descent == "sgd"){
		SGD gd(w, b, y_predict, y, x, y_test, x_test, learning_rate, "mse");
		gd.optimize();
		w = gd.get_weight();
		b = gd.get_bias();
	}
	else{
		std::cout << "Error: specify gradient descent type" << std::endl;
		exit(0);
	}
	
	y_predict = predict();
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
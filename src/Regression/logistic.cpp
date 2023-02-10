#include <iostream>
#include <math.h>
#include <Regression/logistic.h>
#include <Eigen3/Eigen/Dense>
#include <Loss/binary_cross_entropy.h>
#include <Optimizers/bgd.h>
#include <Optimizers/sgd.h>


MatrixXd Logistic::train(MatrixXd x_test, MatrixXd y_test, double learning_rate, std::string gradient_descent, double epsilon, int iterations, int iteration_skip){
	MatrixXd y_predict = predict();
	if (gradient_descent == "bgd"){
		BGD gd(w, b, y_predict, y, x, y_test, x_test, learning_rate, "bce", epsilon, iterations);
		gd.optimize(iteration_skip);
		w = gd.get_weight();
		b = gd.get_bias();
	}
	else if (gradient_descent == "sgd"){
		SGD gd(w, b, y_predict, y, x, y_test, x_test, learning_rate, "bce", epsilon, iterations);
		gd.optimize(iteration_skip);
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

MatrixXd Logistic::predict(){
	MatrixXd y_predict;
	MatrixXd b_vec = MatrixXd::Constant(y.rows(), 1, b); 
	y_predict = x * w + b_vec;
	for (int i = 0; i < y_predict.rows(); i++){
		y_predict(i,0) = 1 / (1 + exp(-y_predict(i,0)));
	}
	return y_predict;
}

MatrixXd Logistic::predict(MatrixXd x_i){
	MatrixXd y_predict;
	MatrixXd b_vec = MatrixXd::Constant(x_i.rows(), 1, b);
	y_predict = x_i * w + b_vec;
	for (int i = 0; i < y_predict.rows(); i++){
		y_predict(i,0) = 1 / (1 + exp(-y_predict(i,0)));
	}
	return y_predict;
}

double Logistic::evaluate(MatrixXd x_test, MatrixXd y_test){
	MatrixXd y_predict = predict(x_test);
	BinaryCrossEntropy bce(y_predict, y_test, x_test);
	return bce.get_error();
}
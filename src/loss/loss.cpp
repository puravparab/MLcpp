#include <iostream>
#include "loss.h"

// mean squared error
float mse(Eigen::MatrixXf y_predict, Eigen::MatrixXf y){
	if (y_predict.size() != y.size()){
		std::cerr << "MSE error: Dimensions do not match" << std::endl;
	}
	auto loss = (y_predict - y).array().square().mean();
	return loss;
}
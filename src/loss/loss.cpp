#include <iostream>
#include "loss.h"

using loss_function = std::function<float(const Eigen::VectorXf& y_predict, const Eigen::VectorXf& y_true)>;

// mean squared error
float mse(const Eigen::VectorXf& y_predict, const Eigen::VectorXf& y_true){
	if (y_predict.size() != y_true.size()){
		std::cerr << "MSE error: Dimensions do not match" << std::endl;
	}
	auto loss = (y_predict - y_true).array().square().mean() / 2;
	return loss;
}
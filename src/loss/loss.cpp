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

// binary cross-entropy error
float bce(const Eigen::VectorXf& y_predict, const Eigen::VectorXf& y_true){
	Eigen::VectorXf epsilon = 1e-15 * Eigen::VectorXf::Ones(y_true.size());
	Eigen::VectorXf clipped_y_pred = y_predict.cwiseMax(epsilon).cwiseMin(Eigen::VectorXf::Ones(y_true.size()) - epsilon);
	return -(
		(y_true.array() * (clipped_y_pred.array().log())).sum() +
		((1.0 - y_true.array()) * ((1.0 - clipped_y_pred.array()).log())).sum()) /
		y_true.size();
}
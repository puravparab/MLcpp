// batch gradient descent
// bgd.h

#ifndef BGD_H
#define BGD_H

#include <Eigen3/Eigen/Dense>
using Eigen::MatrixXd;
#include <Loss/mean_squared_error.h>

class BGD{
	MatrixXd w; // Weights (n rows, 1 column)
	double b; // Bias
	MatrixXd y_predict; // Model Predictions
	MatrixXd y_train; // Training output
	MatrixXd x_train; // Training input

	double learning_rate; // learning rate alpha
	std::string error_type;
	public:
		BGD(MatrixXd weights, double bias, MatrixXd y_predict, MatrixXd y_train, MatrixXd x_train, 
			double learning_rate, std::string error_type)
			: w(weights), b(bias), y_predict(y_predict), y_train(y_train), x_train(x_train), 
				learning_rate(learning_rate), error_type(error_type)
			{}

		MatrixXd update_weights();
		double update_bias();
		MatrixXd get_weight();
		double get_bias();
		void optimize();
};

#endif /* BGD_H */
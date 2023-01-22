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
		MeanSquaredError mse;
	public:
		BGD(MatrixXd weights, double bias, MatrixXd y_predict, MatrixXd y_train, MatrixXd x_train, double learning_rate)
			: w(weights), b(bias), y_predict(y_predict), y_train(y_train), x_train(x_train), learning_rate(learning_rate),
				mse(MeanSquaredError(y_predict, y_train, x_train))
			{}

		MatrixXd update_weights();
		double update_bias();
		void optimize(MatrixXd* new_weights, double* new_bias);
};

#endif /* BGD_H */
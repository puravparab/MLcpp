// stochastic gradient descent
// sgd.h

#ifndef SGD_H
#define SGD_H

#include <Eigen3/Eigen/Dense>
using Eigen::MatrixXd;
#include <Loss/mean_squared_error.h>

class SGD{
	MatrixXd w; // Weights (n rows, 1 column)
	double b; // Bias
	MatrixXd y_predict; // Model Predictions (yhat)
	MatrixXd y_train; // Training output (y)
	MatrixXd x_train; // Training input (x)

	MatrixXd y_test; // Test output
	MatrixXd x_test; // Test input
	
	double learning_rate; // Learning rate (alpha)
	std::string error_type; // Loss function you want to use
	double epsilon; // Minimun error threshold
	int iterations; // Maximum number of iterations
	int size = 100; // Batch size at each iteration of gradient descent

	public:
		SGD(MatrixXd weights, double bias, MatrixXd y_predict, 
			MatrixXd y_train, MatrixXd x_train, 
			MatrixXd y_test, MatrixXd x_test,
			double learning_rate, std::string error_type, 
			double epsilon, int iterations)
			: w(weights), b(bias), y_predict(y_predict), 
				y_train(y_train), x_train(x_train), 
				y_test(y_test), x_test(x_test),
				learning_rate(learning_rate), error_type(error_type), 
				epsilon(epsilon), iterations(iterations)
			{}

		MatrixXd update_weights(MatrixXd x, MatrixXd y, MatrixXd y_prediction);
		double update_bias(MatrixXd x, MatrixXd y, MatrixXd y_prediction);
		MatrixXd get_weight();
		double get_bias();
		void optimize(int iteration_skip);
};

#endif // SGD_H
#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Regression/linear.h>
#include <Loss/mean_squared_error.h>
#include <Optimizers/sgd.h>

MatrixXd Linear::train(double learning_rate){
	MatrixXd y_predict = predict();
	MeanSquaredError mse(y_predict, y, x);
	double prev_error = std::numeric_limits<double>::infinity();

	while (true){
		double current_error = mse.get_error();
		std::cout << current_error << std::endl;
		// If error is minimized
		if (current_error >= prev_error){
			break;
		}
		prev_error = current_error;

		// Run Stochastic gradient descent
		SGD sgd(w, b, y_predict, y, x, learning_rate);
		w(0,0) = sgd.update_weights();
		b = sgd.update_bias();
		y_predict = predict();
		// std::cout << "weights: [" << w.transpose() << "], bias= " << b << std::endl; 
		mse = MeanSquaredError (y_predict, y, x);
	}

	return y_predict;
}

MatrixXd Linear::predict(){
	MatrixXd y = MatrixXd::Constant(x.rows(), 1, 0);
	// Iterate through training examples:
	for (int i = 0; i < x.rows(); i++){
		double value = 0;
		y(i,0) = predict(x.row(i));
	}
	return y;
}

double Linear::predict(MatrixXd x_i){
	double y = 0;
	for (int i = 0; i < w.rows(); i++){
		y += w(i,0) * x_i(0,i);
	}
	y += b;
	return y;
}
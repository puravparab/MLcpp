#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Regression/linear.h>
#include <Loss/mean_squared_error.h>
#include <Optimizers/sgd.h>

double SGD::update_weights(){
	double w_new = w(0,0) - learning_rate * mse.get_derivative_w();
	return w_new;
}

double SGD::update_bias(){
	double b_new = b - learning_rate * mse.get_derivative_b();
	return b_new;
}

void SGD::optimize(MatrixXd* new_weights, double* new_bias){
	MeanSquaredError mse(y_predict, y_train, x_train);
	double prev_error = std::numeric_limits<double>::infinity();
	int count = 0; // Iteration count
	
	while (true){
		double curr_cost = mse.get_error();
		// Print out cost at every 1000th iteration
		if(count % 20000 == 0){
			std::cout << "Iteration step #" << count << ": "<< curr_cost << std::endl;
		}
		
		// If error is minimized
		if (curr_cost >= prev_error || curr_cost < 0.04){
			std::cout << "Iteration step #" << count << ": "<< curr_cost << std::endl;
			break;
		}
		prev_error = curr_cost;

		// Run Stochastic gradient descent
		SGD sgd(w, b, y_predict, y_train, x_train, learning_rate);

		w(0,0) = sgd.update_weights(); // Update weights
		b = sgd.update_bias(); // Update Bias
		Linear linear(x_train, y_train, w, b);
		y_predict = linear.predict();

		count += 1;
		mse = MeanSquaredError (y_predict, y_train, x_train);
	}

	std::cout << std::endl << "Gradient descent steps = " << count << std::endl;
	std::cout << "Weights: [" << w.transpose() << "]" << std::endl;
	std::cout << "Bias: " << b << std::endl;

	// Update weights and bia
	*new_weights = w;
	*new_bias = b;
}
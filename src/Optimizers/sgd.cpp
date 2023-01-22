#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Regression/linear.h>
#include <Loss/mean_squared_error.h>
#include <Optimizers/sgd.h>

MatrixXd SGD::update_weights(){
	MatrixXd w_new = w - (learning_rate * (1.0 / y_predict.rows()) * mse.get_derivative_w());
	return w_new;
}

double SGD::update_bias(){
	double b_new = b - (learning_rate * (1.0 / y_predict.rows()) * mse.get_derivative_b());
	return b_new;
}

MatrixXd SGD::get_weight(){
	return w;
}

double SGD::get_bias(){
	return b;
}

void SGD::optimize(){
	MeanSquaredError mse(y_predict, y_train, x_train);
	double prev_error = std::numeric_limits<double>::infinity();
	int count = 0; // Iteration count

	while (true){
		double curr_cost = mse.get_error();
		// Print out cost at every 1000th iteration
		if(count % 20000 == 0){
			std::cout << "Step #" << count << ": Cost = "<< curr_cost << std::endl;
		}
		
		// If error is minimized
		if (curr_cost >= prev_error || curr_cost < 0.02){
			std::cout << "Step #" << count << ": Cost="<< curr_cost << std::endl;
			break;
		}
		prev_error = curr_cost;

		// Run Stochastic gradient descent
		w = update_weights(); // Update weights
		b = update_bias(); // Update Bias
		Linear linear(x_train, y_train, w, b, "sgd");
		y_predict = linear.predict();

		count += 1;
		mse = MeanSquaredError (y_predict, y_train, x_train);
	}

	std::cout << std::endl << "Gradient descent steps = " << count << std::endl;
	std::cout << "Weights: [" << w.transpose() << "]" << std::endl;
	std::cout << "Bias: " << b << std::endl;
}
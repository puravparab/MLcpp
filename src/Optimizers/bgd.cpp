#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Regression/linear.h>
#include <Regression/logistic.h>
#include <Loss/mean_squared_error.h>
#include <Loss/binary_cross_entropy.h>
#include <Optimizers/bgd.h>

MatrixXd BGD::update_weights(){
	MatrixXd w_new;
	if (error_type == "mse"){
		MeanSquaredError mse(y_predict, y_train, x_train);
		w_new = w - (learning_rate * (1.0 / y_predict.rows()) * mse.get_derivative_w());
	} 
	// Using binary cross entropy:
	else if (error_type == "bce"){
		BinaryCrossEntropy bce(y_predict, y_train, x_train);
		w_new = w - (learning_rate * (1.0 / y_predict.rows()) * bce.get_derivative_w());
	}
	return w_new;
}

double BGD::update_bias(){
	double b_new;
	if (error_type == "mse"){
		MeanSquaredError mse(y_predict, y_train, x_train);
		b_new = b - (learning_rate * (1.0 / y_predict.rows()) * mse.get_derivative_b());
	} 
	// Using binary cross entropy:
	else if (error_type == "bce"){
		BinaryCrossEntropy bce(y_predict, y_train, x_train);
		b_new = b - (learning_rate * (1.0 / y_predict.rows()) * bce.get_derivative_b());
	}
	return b_new;
}

MatrixXd BGD::get_weight(){
	return w;
}
double BGD::get_bias(){
	return b;
}

void BGD::optimize(){
	double prev_error = std::numeric_limits<double>::infinity();
	int count = 0; // Iteration count

	double curr_cost = 0;
	// Using mean squared error:
	if (error_type == "mse"){
		MeanSquaredError mse(y_predict, y_train, x_train);
		curr_cost = mse.get_error();
	} 
	// Using binary cross entropy:
	else if (error_type == "bce"){
		BinaryCrossEntropy bce(y_predict, y_train, x_train);
		curr_cost = bce.get_error();
	}

	while (abs(prev_error - curr_cost) > epsilon && count <= iterations){
		// Print count at every iterval
		if(count % 200 == 0){
			std::cout << "Step #" << count << ": Cost = "<< curr_cost << std::endl;
		}

		prev_error = curr_cost;

		// Run Stochastic gradient descent
		w = update_weights(); // Update weights
		b = update_bias(); // Update Bias

		// Using mean squared error:
		if (error_type == "mse"){
			Linear linear(x_train, y_train, w, b);
			y_predict = linear.predict();
			MeanSquaredError mse(y_predict, y_train, x_train);
			curr_cost = mse.get_error();
		} 
		// Using binary cross entropy:
		else if (error_type == "bce"){
			Logistic logistic(x_train, y_train, w, b);
			y_predict = logistic.predict();
			BinaryCrossEntropy bce(y_predict, y_train, x_train);
			curr_cost = bce.get_error();
		}
		count += 1;
	}
	std::cout << "Step #" << count << ": Cost = "<< curr_cost << std::endl;

	std::cout << std::endl << "Gradient descent steps = " << count << std::endl;
	std::cout << "Weights: [" << w.transpose() << "]" << std::endl;
	std::cout << "Bias: " << b << std::endl;
}
#include <iostream>
#include <ctime>
#include <Eigen3/Eigen/Dense>
#include <Regression/linear.h>
#include <Loss/mean_squared_error.h>
#include <Loss/binary_cross_entropy.h>
#include <Optimizers/sgd.h>

MatrixXd SGD::update_weights(MatrixXd x, MatrixXd y, MatrixXd y_prediction){
	MatrixXd w_new;
	if (error_type == "mse"){
		MeanSquaredError mse(y_prediction, y, x);
		w_new = w - (learning_rate * (1.0 / y_predict.rows()) * mse.get_derivative_w());
	} 
	// Using binary cross entropy:
	else if (error_type == "bce"){
		BinaryCrossEntropy bce(y_prediction, y, x);
		w_new = w - (learning_rate * (1.0 / y_predict.rows()) * bce.get_derivative_w());
	}
	
	return w_new;
}

double SGD::update_bias(MatrixXd x, MatrixXd y, MatrixXd y_prediction){
	double b_new;
	if (error_type == "mse"){
		MeanSquaredError mse(y_prediction, y, x);
		b_new = b - (learning_rate * (1.0 / y.rows()) * mse.get_derivative_b());
	} 
	// Using binary cross entropy:
	else if (error_type == "bce"){
		BinaryCrossEntropy bce(y_prediction, y, x);
		 b_new = b - (learning_rate * (1.0 / y.rows()) * bce.get_derivative_b());
	}
	return b_new;
}

MatrixXd SGD::get_weight(){
	return w;
}
double SGD::get_bias(){
	return b;
}

void SGD::optimize(){
	srand(time(0));
	double prev_error = std::numeric_limits<double>::infinity();
	int count = 0; // Iteration count
	double epsilon = 1e-1; // Maximum convergence difference
	int iteration = 90000; // Max iterations allowed
	int size = 1; // Size of the sample at each iteration

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

	// Run stochastic gradient descent
	while (true){
		// Print count at every iterval
		if(count % 10000 == 0){
			std::cout << "Step #" << count << ": Cost = "<< curr_cost << std::endl;
		}
		
		// If error is minimized
		if (abs(prev_error - curr_cost) <= epsilon || count > iteration){
			std::cout << "Step #" << count << ": Cost="<< curr_cost << std::endl;
			break;
		}
		prev_error = curr_cost;

		// Create matrices for random sample
		MatrixXd x_gd(size, x_train.cols());
		MatrixXd y_gd(size, y_train.cols());
		MatrixXd y_predict_gd(size, y_train.cols());
		// Assign random sample
		for (int i = 0; i < size; i++){
			int index = rand() % x_train.rows();
			x_gd.row(i) = x_train.row(index);
			y_gd.row(i) = y_train.row(index);
			y_predict_gd.row(i) = y_predict.row(index);
		}

		// Run Stochastic gradient descent and update parameters
		w = update_weights(x_gd, y_gd, y_predict_gd); // Update weights
		b = update_bias(x_gd, y_gd, y_predict_gd); // Update Bias
		
		Linear linear(x_test, y_test, w, b);
		MatrixXd y_predict_test = linear.predict();

		// Using mean squared error:
		if (error_type == "mse"){
			MeanSquaredError mse(y_predict_test, y_test, x_test);
			curr_cost = mse.get_error();
		} 
		// Using binary cross entropy:
		else if (error_type == "bce"){
			BinaryCrossEntropy bce(y_predict_test, y_test, x_test);
		}

		count += 1;
	}

	std::cout << std::endl << "Gradient descent steps = " << count << std::endl;
	std::cout << "Weights: [" << w.transpose() << "]" << std::endl;
	std::cout << "Bias: " << b << std::endl;
}
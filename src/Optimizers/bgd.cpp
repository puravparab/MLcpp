#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Regression/linear.h>
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
	while (true){
		// Print out cost at every 1000th iteration
		if(count % 200 == 0){
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
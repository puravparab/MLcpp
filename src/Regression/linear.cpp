#include <iostream>
#include <Eigen3/Eigen/Dense>
#include <Regression/linear.h>
#include <Loss/mean_squared_error.h>
#include <Optimizers/sgd.h>

MatrixXd Linear::train(double learning_rate){
	MatrixXd y_predict = predict();
	MeanSquaredError mse(y_predict, y, x);

	double prev_error = std::numeric_limits<double>::infinity();
	int count = 0; // Iteration count

	while (true){
		double curr_cost = mse.get_error();
		// Print out cost at every 1000th iteration
		if(count % 1000 == 0){
			std::cout << "Iteration step #" << count << ": "<< curr_cost << std::endl;
		}
		
		// If error is minimized
		if (curr_cost >= prev_error || curr_cost < 0.05){
			break;
		}
		prev_error = curr_cost;

		// Run Stochastic gradient descent
		SGD sgd(w, b, y_predict, y, x, learning_rate);

		w(0,0) = sgd.update_weights(); // Update weights
		b = sgd.update_bias(); // Update Bias
		y_predict = predict();
		
		count += 1;
		mse = MeanSquaredError (y_predict, y, x);
	}

	std::cout << std::endl << "Gradient descent steps = " << count << std::endl;
	std::cout << "Weights: [" << w.transpose() << "]" << std::endl;
	std::cout << "Bias: " << b << std::endl;

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
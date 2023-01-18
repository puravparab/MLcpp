#include <iostream>
#include <cmath>
#include <Eigen3/Eigen/Dense>
#include <Loss/mean_squared_error.h>

/* Mean Squared Error:

	J(w,b) = (1/2n) x sum[1,n](y_predict[i] - y_train[i])^2

	where:
	 n = number of training examples
	 i = ith training example
*/

double MeanSquaredError::get_error(){
	double error = 0;
	// Iterate through predictions
	for (int i = 0; i < y_predict.rows(); i++){
		error += pow((y_predict(i,0) - y_train(i,0)), 2);
	}
	return (1.0/(2.0*y_predict.rows())) * error;
}

// Does not work with multiple features
double MeanSquaredError::get_derivative_w(){
	double error = 0;
	// Iterate through training examples
	for (int i = 0; i < y_predict.rows(); i++){
		// Iterate through features
		for (int j = 0; j < y_predict.cols(); j++){
			error += (y_predict(i,j) - y_train(i,j)) * x_train(i,j) ;
		}
	}
	return (1.0/y_predict.rows()) * error;
}

// Does not work with multiple features
double MeanSquaredError::get_derivative_b(){
	double error = 0;
	// Iterate through training examples
	for (int i = 0; i < y_predict.rows(); i++){
		// Iterate through features
		for (int j = 0; j < y_predict.cols(); j++){
			error += (y_predict(i,j) - y_train(i,j));
		}
	}
	return (1.0/y_predict.rows()) * error;
}
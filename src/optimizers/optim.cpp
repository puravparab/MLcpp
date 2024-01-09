#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include "optim.h"

// gradient descent function:
// @param X: matrix (examples x (params + 1)) (bias column in the end)
// @param Y: vector containing true values
// @param W: vector of weights (includes bias term at the end)
// @param lr: learning rate
// @returns gd: vector of updated weights and bias
Eigen::VectorXf gd(const Eigen::MatrixXf& X, const Eigen::VectorXf& Y, Eigen::VectorXf& W, float lr){
  Eigen::VectorXf y_pred = X * W;
  Eigen::VectorXf error = y_pred - Y;
  Eigen::VectorXf gradient = (X.transpose() * error) / X.rows();
  W = W - (gradient * lr);
  return W;
}

// full batch gradient descent:
// run gradient descent on all training examples
// @param X: matrix (examples x params)
// @param Y: vector containing true values
// @param W: vector of weights
// @param B: bias term
// @param lr: learning rate
// @param loss_function: loss_function from "loss/loss.h"
// @return: a vector containing weights, bias and loss
Eigen::VectorXf batchgd(const Eigen::MatrixXf& X, const Eigen::VectorXf& Y, Eigen::VectorXf& W, float B, float lr, loss_function loss_function){
  Eigen::MatrixXf X_(X.rows(), X.cols() + 1);
  Eigen::VectorXf W_(X.cols() + 1);
  X_ << X, Eigen::VectorXf::Constant(X.rows(), B);
  W_ << W, B;

  W_ = gd(X_, Y, W_, lr); // get updated weights and bias
  Eigen::VectorXf Y_pred = X_ * W_; // get predictions
  float loss = loss_function(Y_pred, Y); // get new loss

  Eigen::VectorXf history(W_.rows() + 1); // capture weights, bias, loss
  history << W_, loss;
  return history;
}

// batch gradient descent for logistic regression
Eigen::VectorXf batchgd_logistic(const Eigen::MatrixXf& X, const Eigen::VectorXf& Y, Eigen::VectorXf& W, float B, float lr, loss_function loss_function){
  Eigen::MatrixXf X_(X.rows(), X.cols() + 1);
  Eigen::VectorXf W_(X.cols() + 1);
  X_ << X, Eigen::VectorXf::Constant(X.rows(), B);
  W_ << W, B;

  W_ = gd(X_, Y, W_, lr); // get updated weights and bias
  Eigen::VectorXf Y_pred = X_ * W_; // get predictions
	Y_pred = 1.0 / (1.0 + (-Y_pred.array()).exp()); // get sigmoid
  float loss = loss_function(Y_pred, Y); // get new loss

  Eigen::VectorXf history(W_.rows() + 1); // capture weights, bias, loss
  history << W_, loss;
  return history;
}

// mini-batch gradient descent:
// run gradient descent on batches
// @param X: matrix (examples x params)
// @param Y: vector containing true values
// @param W: vector of weights
// @param B: bias term
// @param lr: learning rate
// @param loss_function: loss_function from "loss/loss.h"
// @param batch: loss_function from "loss/loss.h"
// @return: a vector containing weights, bias and loss
Eigen::MatrixXf mbgd(const Eigen::MatrixXf& X, const Eigen::VectorXf& Y, Eigen::VectorXf& W, float B, float lr, loss_function loss_function, int batch){
	int rows = X.rows();
	if (batch > rows){
		std::cerr << "Error: batch size should be less that no of examples" << std::endl;
	}

	Eigen::MatrixXf X_(X.rows(), X.cols() + 1);
  Eigen::VectorXf W_(X.cols() + 1);
  X_ << X, Eigen::VectorXf::Constant(X.rows(), B);
  W_ << W, B;

	int iterations = rows / batch;
	Eigen::MatrixXf history(iterations, W_.rows() + 1);
	for (int i = 0; i < iterations; i++){
		int start = i * batch;
		Eigen::MatrixXf x_subset = X_.block(start, 0, batch, X_.cols());
		Eigen::VectorXf y_subset = Y.segment(start, batch);

		W_ = gd(x_subset, y_subset, W_, lr); // get updated weights and bias
		Eigen::VectorXf Y_pred = X_ * W_; // get predictions
		float loss = loss_function(Y_pred, Y); // get new loss

		Eigen::VectorXf iter_history(W_.rows() + 1); // capture weights, bias, loss
		iter_history << W_, loss;
		history.row(i) = iter_history;
	}
  return history;
}

// stochastic gradient descent:
// @param X: matrix (examples x params)
// @param Y: vector containing true values
// @param W: vector of weights
// @param B: bias term
// @param lr: learning rate
// @param loss_function: loss_function from "loss/loss.h"
// @param batch: loss_function from "loss/loss.h"
// @return: a vector containing weights, bias and loss
Eigen::MatrixXf sgd(const Eigen::MatrixXf& X, const Eigen::VectorXf& Y, Eigen::VectorXf& W, float B, float lr, loss_function loss_function, int batch){
	int rows = X.rows();
	if (batch > rows){
		std::cerr << "Error: batch size should be less that no of examples" << std::endl;
	}

	Eigen::MatrixXf X_(X.rows(), X.cols() + 1);
  Eigen::VectorXf W_(X.cols() + 1);
  X_ << X, Eigen::VectorXf::Constant(X.rows(), B);
  W_ << W, B;

	// shuffle dataset
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine rng(seed);
	std::vector<int> indices(rows);
	std::iota(indices.begin(), indices.end(), 0);
	std::shuffle(indices.begin(), indices.end(), rng);
	Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(rows);
	perm.indices() = Eigen::Map<Eigen::VectorXi>(indices.data(), indices.size());
	X_ = perm * X_;
	Eigen::VectorXf Y_ = perm * Y;

	int iterations = rows / batch;
	Eigen::MatrixXf history(iterations, W_.rows() + 1);
	for (int i = 0; i < iterations; i++){
		int start = i * batch;
		Eigen::MatrixXf x_subset = X_.block(start, 0, batch, X_.cols());
		Eigen::VectorXf y_subset = Y.segment(start, batch);

		W_ = gd(x_subset, y_subset, W_, lr); // get updated weights and bias
		Eigen::VectorXf Y_pred = X_ * W_; // get predictions
		float loss = loss_function(Y_pred, Y); // get new loss

		Eigen::VectorXf iter_history(W_.rows() + 1); // capture weights, bias, loss
		iter_history << W_, loss;
		history.row(i) = iter_history;
	}
  return history;
}
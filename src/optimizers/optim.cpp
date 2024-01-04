#include <iostream>
#include "optim.h"

// gradient descent function:
// X: matrix (examples x (params + 1)) (bias column in the end)
// Y: vector containing true values
// W: vector of weights (includes bias term at the end)
// lr: learning rate
Eigen::VectorXf gd(const Eigen::MatrixXf& X, const Eigen::VectorXf& Y, Eigen::VectorXf& W, float lr){
  Eigen::VectorXf y_pred = X * W;
  Eigen::VectorXf error = y_pred - Y;
  Eigen::VectorXf gradient = (X.transpose() * error) / X.rows();
  W = W - (gradient * lr);
  return W;
}

// full batch gradient descent:
// run gradient descent on all training examples
// X: matrix (examples x params)
// Y: vector containing true values
// W: vector of weights
// B: bias term
// lr: learning rate
// loss_function: loss_function from "loss/loss.h"
Eigen::VectorXf batchgd(const Eigen::MatrixXf& X, const Eigen::VectorXf& Y, Eigen::VectorXf& W, float B, float lr, loss_function loss_function){
  Eigen::MatrixXf X_(X.rows(), X.cols() + 1);
  Eigen::VectorXf W_(X.cols() + 1);
  X_ << X, Eigen::VectorXf::Constant(X.rows(), B);
  W_ << W, B;

  // get updated weights and bias
  W_ = gd(X_, Y, W_, lr);

  // get predictions
  Eigen::VectorXf Y_pred = X_ * W_;

  // get new loss
  float loss = loss_function(Y_pred, Y);
  
  Eigen::VectorXf history(W_.rows() + 1); // capture weights, loss
  history << W_, loss;
  return history;
}
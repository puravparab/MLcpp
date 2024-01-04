#ifndef OPTIM_H
#define OPTIM_H

#include <vector>
#include <functional>
#include <Eigen/Dense>

using loss_function = std::function<float(const Eigen::VectorXf&, const Eigen::VectorXf&)>;

// gradient descent function
Eigen::VectorXf gd(const Eigen::MatrixXf& X, const Eigen::VectorXf& Y, Eigen::VectorXf& W, float lr);

// full batch gradient descent
Eigen::VectorXf batchgd(const Eigen::MatrixXf& X, const Eigen::VectorXf& Y, Eigen::VectorXf& W, float B, float lr, loss_function loss_function);
#endif // OPTIM_H
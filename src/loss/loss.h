#ifndef LOSS_H
#define LOSS_H

#include <Eigen/Dense>

float mse(const Eigen::VectorXf& y_predict, const Eigen::VectorXf& y_true);

#endif // LOSS_H
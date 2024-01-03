#ifndef LOSS_H
#define LOSS_H

#include <Eigen/Dense>

float mse(Eigen::MatrixXf y_predict, Eigen::MatrixXf y);

#endif // LOSS_H
#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>

void save_weights(Eigen::VectorXf& model_weights, int param_count, std::string save_type, std::string filename);
Eigen::VectorXf load_weights(const std::string& filename);

#endif // UTILS_H
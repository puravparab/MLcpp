// Dataset Class
// dataset.h
#ifndef DATASET_H
#define DATASET_H

#include <Eigen3/Eigen/Dense>
using Eigen::MatrixXd;

class Dataset{
	MatrixXd x_train;
	MatrixXd y_train;
	public:
		Dataset(std::string url);
		MatrixXd get_x_train();
		MatrixXd get_y_train();
};

#endif /* DATASET_H */
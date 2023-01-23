// Dataset Class
// dataset.h
#ifndef DATASET_H
#define DATASET_H

#include <Eigen3/Eigen/Dense>
using Eigen::MatrixXd;

class Dataset{
	MatrixXd x_train;
	MatrixXd y_train;
	MatrixXd x_test;
	MatrixXd y_test;
	double split = 100; // Train/test split
	public:
		Dataset(std::string url);
		Dataset(std::string url, double split);
		MatrixXd get_x_train();
		MatrixXd get_x_test();
		MatrixXd get_y_train();
		MatrixXd get_y_test();
};

#endif /* DATASET_H */
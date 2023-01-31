// Dataset Class
// dataset.h
#ifndef DATASET_H
#define DATASET_H

#include <Eigen3/Eigen/Dense>
using Eigen::MatrixXd;

class Dataset{
	MatrixXd train_data;
	MatrixXd test_data;
	double split = 100; // Train/test split
	public:
		Dataset(std::string url);
		Dataset(std::string url, double split);
		MatrixXd get_train();
		MatrixXd get_test();
};

#endif /* DATASET_H */
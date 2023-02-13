// Normalization
// normalization.h

#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include <Eigen3/Eigen/Dense>
using Eigen::MatrixXd;

class Normalization{
	MatrixXd x; // Normalized training set (n examples x m features)

	MatrixXd min;
	MatrixXd max;
	MatrixXd mean;
	MatrixXd std_dev;

	// "mn" = mean normalized
	// "zs" = zscore normalization
	std::string type; // Type of normalization

	public:
		Normalization(MatrixXd x_train);
		Normalization(MatrixXd x_train, std::string type);
		void process();
		MatrixXd process(MatrixXd x_i);
		MatrixXd get_x_train();
};

#endif /* NORMALIZATION_H */
// Normalization
// normalization.h

#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include <Eigen3/Eigen/Dense>
using Eigen::MatrixXd;

class Normalization{
	MatrixXd x; // n examples x m features
	MatrixXd values; // 3 columns x m features

	public:
		Normalization(MatrixXd x_train);
		void process();
		MatrixXd process(MatrixXd x_i);
		MatrixXd get_x_train();
};

#endif /* NORMALIZATION_H */
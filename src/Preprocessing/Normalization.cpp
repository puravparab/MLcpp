#include <iostream>
#include <Preprocessing/normalization.h>

Normalization::Normalization(MatrixXd x_train){
	x = x_train;
	values = MatrixXd(3, x.cols());
	process();
}

void Normalization::process(){
	// Mean Normalization
	for(int i = 0; i < x.cols(); i++){
		double min = x.col(i).minCoeff();
		double max = x.col(i).maxCoeff();
		double mean = x.col(i).mean();
		values(0,i) = mean;
		values(1,i) = max;
		values(2,i) = min;
	}
	x = process(x);
}

MatrixXd Normalization::process(MatrixXd x_i){
	for(int i = 0; i < x_i.rows(); i++){
		for(int j = 0; j < x_i.cols(); j++){
			// normalized x = (x - mean) / (max - min)
			x_i(i,j) = (x_i(i,j) - values(0,j)) / (values(1,j) - values(2,j));
		}
	}
	return x_i;
}

MatrixXd Normalization::get_x_train(){
	return x;
}
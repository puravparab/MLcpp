#include <iostream>
#include <Preprocessing/normalization.h>

Normalization::Normalization(MatrixXd x_train): x(x_train), type("mn"){
	min = MatrixXd::Zero(1, x.cols());
	max = MatrixXd::Zero(1, x.cols());
	mean = MatrixXd::Zero(1, x.cols());
	std_dev = MatrixXd::Zero(1, x.cols());
	process();
}

Normalization::Normalization(MatrixXd x_train, std::string type): x(x_train), type(type){
	min = MatrixXd::Zero(1, x.cols());
	max = MatrixXd::Zero(1, x.cols());
	mean = MatrixXd::Zero(1, x.cols());
	std_dev = MatrixXd::Zero(1, x.cols());
	process();
}

void Normalization::process(){
	for(int i = 0; i < x.cols(); i++){
		min(0,i) = x.col(i).minCoeff();
		max(0,i) = x.col(i).maxCoeff();
		mean(0,i) = x.col(i).mean();
		std_dev(0,i) = sqrt((x.col(i).array() - mean(0,i)).square().sum() / x.col(i).rows());
	}
	x = process(x);
}

MatrixXd Normalization::process(MatrixXd x_i){
	// Mean Normalization
	if (type == "mn"){
		for (int i = 0; i < x.cols(); i++){
			// normalized x = (x - mean) / (max - min)
			x_i.col(i) = (x_i.col(i).array() - mean(0,i)) / (max(0,i) - min(0,i));
		}
	}
	// Z-score Normalization
	if (type == "zs"){
		for (int i = 0; i < x.cols(); i++){
			// normalized x = (x - mean) / (max - min)
			x_i.col(i) = (x_i.col(i).array() - mean(0,i)) / std_dev(0,i);
		}
	}
	return x_i;
}

MatrixXd Normalization::get_x_train(){
	return x;
}
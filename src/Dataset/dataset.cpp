#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <vector>
#include <Dataset/dataset.h>
#include <Eigen3/Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Datasets should have no spaces.
// First row should be categories.
// Last column should be the category being predicted.
Dataset::Dataset(std::string url){
	std::ifstream file(url);
	std::string line, value;
	int no_of_rows = 0;
	int no_of_cols = 0;
	int index = 0;
	if (file.is_open()){
		while(std::getline(file,line)){
			std::stringstream s(line);
			no_of_cols = 0;
			while(std::getline(s, value, ',')){
				no_of_cols += 1;
			}
			// Ignore first line
			if (index != 0){
				no_of_rows += 1;
			}
			index += 1;
		}
	} else{
		std::cout << "Error: File cannot be found" << std::endl;
		exit(0);
	}
	file.close();

	if (no_of_cols < 2){
		std::cout << "Error: Dataset not valid" << std::endl;
	}

	std::cout << "Loading dataset ...\n" <<
	"Rows = " << no_of_rows << std::endl <<
	"Cols = " << no_of_cols << std::endl;

	// Assign values to x_train and y_train
	std::ifstream file1(url);
	MatrixXd data(no_of_rows, no_of_cols);
	line = "";
	value = "";
	std::vector<std::string> row_vec;
	if (file1.is_open()){
		index = 0;
		while(std::getline(file1,line)){
			std::stringstream s(line);
			row_vec.clear();
			while(std::getline(s, value, ',')){
				row_vec.push_back(value);
			}
			// Ignore first line
			if (index > 0){
				for (int i = 0; i < row_vec.size(); i++){
					data(index-1, i) = std::stod(row_vec[i]);
				}
			}
			index += 1;
		}
	} else{
		std::cout << "Error: File cannot be found" << std::endl;
		exit(0);
	}
	train_data = data;
}

Dataset::Dataset(std::string url, double split){
	split = split;
	Dataset temp(url);
	MatrixXd data = temp.get_train();

	int train_rows = floor((split/100) * data.rows()); 
	int test_rows = data.rows() - train_rows;
	train_data = MatrixXd(train_rows, data.cols());
	test_data = MatrixXd(test_rows, data.cols());

	// Should be okay for large datasets
	// Ideally you would want unique random numbers
	srand(time(0));
	// Assign values to training set
	for (int i = 0; i < train_rows; i++){
		int index = rand() % data.rows();
		train_data.row(i) = data.row(index);
	}
	// Assign values to test set
	for (int i = 0; i < test_rows; i++){
		int index = rand() % data.rows();
		test_data.row(i) = data.row(index);
	}
}

MatrixXd Dataset::get_train(){
	return train_data;
}
MatrixXd Dataset::get_test(){
	return test_data;
}
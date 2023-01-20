#include <iostream>
#include <fstream>
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

	// Assign values to x_train and y_train
	std::ifstream file1(url);
	MatrixXd x(no_of_rows, no_of_cols - 1);
	MatrixXd y(no_of_rows, 1);
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
					if (i == row_vec.size() - 1){
						y(index-1, 0) = std::stod(row_vec[i]);
					} else{
						x(index-1, i) = std::stod(row_vec[i]);
					}			
				}
			}
			index += 1;
		}
	} else{
		std::cout << "Error: File cannot be found" << std::endl;
		exit(0);
	}
	x_train = x;
	y_train = y;
}

MatrixXd Dataset::get_x_train(){
	return x_train;
}

MatrixXd Dataset::get_y_train(){
	return y_train;
}
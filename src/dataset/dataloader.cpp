#include <iostream>
#include <algorithm>
#include <random>
#include "dataloader.h"

using dataType = std::variant<std::string, float>;

Dataloader::Dataloader(Dataset& dataset) : dataset(dataset){
	y_index = dataset.get_output_column();

	// move output header to end of headers vector
	auto headers_temp = dataset.get_headers();
	headers_temp.erase(headers_temp.begin() + y_index);
	headers_temp.push_back(dataset.get_headers()[y_index]);
	headers = headers_temp;

	// move output column to end of column summary vector
	auto col_sum_temp = dataset.get_column_summary();
	col_sum_temp.erase(col_sum_temp.begin() + y_index);
	col_sum_temp.push_back(dataset.get_column_summary()[y_index]);
	column_summary = col_sum_temp;
}

std::vector<std::vector<Eigen::MatrixXf>> Dataloader::split(u_int8_t train_percent, u_int8_t test_percent){
	auto data = dataset.get_data();
  uint32_t total_rows = data.size();
  uint32_t train_rows = total_rows * train_percent / 100;
  uint32_t test_rows = total_rows * test_percent / 100;
	uint16_t cols = data[0].size();
  // Shuffle the data randomly
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(data.begin(), data.end(), g);

  // Create training set
  Eigen::MatrixXf x_train(train_rows, cols - 1);
	Eigen::MatrixXf y_train(train_rows, 1);
  for (uint32_t i = 0; i < train_rows; ++i) {
		uint16_t xcol_index = 0;
		for (uint16_t j = 0; j < cols; ++j){
			try{
				auto test = std::get<float>(data[i][j]);
			} catch(const std::exception& e){
				std::cerr << e.what() << '\n';
				exit(1);
			}
			
			if (j == y_index){
				y_train(i, 0) = std::get<float>(data[i][j]);
			} else {
				x_train(i, xcol_index) = std::get<float>(data[i][j]);
				xcol_index++;
			}
		}
  }

  // Create testing set
  Eigen::MatrixXf x_test(test_rows, cols - 1);
	Eigen::MatrixXf y_test(test_rows, 1);
	for (uint32_t i = 0 ; i < test_rows; ++i) {
		uint16_t xcol_index = 0;
		for (uint16_t j = 0; j < cols; ++j){
			try{
				auto test = std::get<float>(data[i][j]);
			} catch(const std::exception& e){
				std::cerr << e.what() << '\n';
				exit(1);
			}

			if (j == y_index){
				y_test(i, 0) = std::get<float>(data[i + train_rows][j]);
			} else {
				x_test(i, xcol_index) = std::get<float>(data[i + train_rows][j]);
				xcol_index++;
			}
		}
  }

	// Calculate min, max, mean, std_dev
	Eigen::MatrixXf data_matrix(x_train.rows() + x_test.rows(), x_train.cols());
	data_matrix << x_train, x_test;
	min = data_matrix.colwise().minCoeff();
  max = data_matrix.colwise().maxCoeff();
	mean = data_matrix.colwise().mean();
	std_dev = ((data_matrix.rowwise() - mean.transpose()).array().square().colwise().sum() / (data_matrix.rows() - 1)).sqrt();

  return std::vector<std::vector<Eigen::MatrixXf>> {{x_train, y_train}, {x_test, y_test}};
}
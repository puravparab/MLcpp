#include <iostream>
#include <algorithm>
#include <random>
#include "dataloader.h"

using dataType = std::variant<std::string, float>;

Dataloader::Dataloader(Dataset& dataset) : dataset(dataset){
	y_index = dataset.get_output_column();
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
			if (j == y_index){
				y_test(i, 0) = std::get<float>(data[i + train_rows][j]);
			} else {
				x_test(i, xcol_index) = std::get<float>(data[i + train_rows][j]);
				xcol_index++;
			}
		}
  }
  return std::vector<std::vector<Eigen::MatrixXf>> {{x_train, y_train}, {x_test, y_test}};
}
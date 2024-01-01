#include <algorithm>
#include <random>
#include "dataloader.h"

using dataType = std::variant<std::string, float>;

Dataloader::Dataloader(std::vector<std::vector<dataType>>& data) : data(data){}

std::vector<Eigen::MatrixXf> Dataloader::split(u_int8_t train_percent, u_int8_t test_percent){
  // Assuming train_percent + test_percent equals 100
  uint32_t total_rows = data.size();
  uint32_t train_rows = total_rows * train_percent / 100;
  uint32_t test_rows = total_rows * test_percent / 100;
	uint16_t cols = data[0].size();
  // Shuffle the data randomly
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(data.begin(), data.end(), g);

  // Create training set
  Eigen::MatrixXf train_data(train_rows, cols);
  for (uint32_t i = 0; i < train_rows; ++i) {
		for (uint16_t j = 0; j < cols; ++j){
			train_data(i, j) = std::get<float>(data[i][j]);
		}
  }
  // Create testing set
  Eigen::MatrixXf test_data(test_rows, cols);
	for (uint32_t i = 0 ; i < test_rows; ++i) {
		for (uint16_t j = 0; j < cols; ++j){
			test_data(i, j) = std::get<float>(data[i + train_rows][j]);
		}
  }
  return {train_data, test_data};
}
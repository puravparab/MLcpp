#ifndef DATALOADER_H
#define DATALOADER_H

#include <cstdint>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include "dataset.h"

using dataType = std::variant<std::string, float>;

class Dataloader{
	public:
		Dataloader(Dataset& dataset);
		Dataset& dataset;
		uint16_t y_index;
		std::vector<std::string> headers;
		std::vector<Column_Summary> column_summary;
		Eigen::VectorXf min, max, mean, std_dev;
		std::vector<std::vector<Eigen::MatrixXf>> split(u_int8_t train_percent, u_int8_t test_percent);
		// std::vector<Eigen::MatrixXf> split(u_int8_t train_percent, u_int8_t test_percent, u_int8_t validation_percent);
};

#endif // DATALOADER_H
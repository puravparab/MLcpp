#ifndef DATALOADER_H
#define DATALOADER_H

#include <cstdint>
#include <vector>
#include <string>
#include <Eigen/Dense>

using dataType = std::variant<std::string, float>;

class Dataloader{
	public:
		Dataloader(std::vector<std::vector<dataType>>& data);
		std::vector<std::vector<dataType>>& data;
		std::vector<Eigen::MatrixXf> split(u_int8_t train_percent, u_int8_t test_percent);
		// std::vector<Eigen::MatrixXf> split(u_int8_t train_percent, u_int8_t test_percent, u_int8_t validation_percent);
};

#endif // DATALOADER_H
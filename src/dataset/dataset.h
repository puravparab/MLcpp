#ifndef DATASET_H
#define DATASET_H

#include <cstdint>
#include <typeindex>
#include <vector>
#include <string>
#include <Eigen/Dense>

class Header_Item {
	public:
		std::string name;
		std::type_index type;
		Header_Item(std::string name, std::type_index type);
};
class Dataset {
	public:
		Dataset();
		void read(const std::string& file_path);
		const std::vector<uint32_t> shape();
		const void head(uint8_t n);
		// One hot encoding function
		// Drop column function
	private:
		Eigen::MatrixXf data;
		uint32_t length;
		uint16_t col_length; // Number of columns
		uint16_t y_index; // Index of the training output
		std::vector<Header_Item> headers;
};

#endif // DATASET_H
#ifndef DATASET_H
#define DATASET_H

#include <cstdint>
#include <typeindex>
#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>

using dataType = std::variant<std::string, float>;

class Column_Summary {
	public:
		std::string name;
		std::type_index type;
		float mean, sum, std_dev, max, min;
		std::vector<uint32_t> null_index;
		std::unordered_map<std::string, int>unique_strings;
		Column_Summary(std::string name);	
};

// 2D Dataset
class Dataset {
	public:
		Dataset();
		void read(const std::string file_path);
		const std::vector<uint32_t> shape();
		const void head(uint8_t n, int width);
		const void print_headers();
		const void col_summary(std::string name);
		// Drop column function
		// One hot encoding
	private:
		std::vector<std::vector<dataType>> data;
		uint32_t length;
		uint16_t col_length; // Number of columns
		uint16_t y_index; // Index of the training output
		std::vector<Column_Summary> column_summary;

		void update_header_type();
		void handle_null_values();
		void summarize_columns();
};

#endif // DATASET_H
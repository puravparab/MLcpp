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
		void set_output_column(std::string name);
		uint16_t get_output_column();
		const std::vector<uint32_t> shape();
		const void head(uint8_t n, int width);
		std::vector<std::string> get_headers();
		const void print_headers();
		const void col_summary(std::string name);
		void drop_column(std::string name);
		void drop_null_rows();
		void one_hot_encoding(std::string name);
		std::vector<std::vector<dataType>> get_data();
	private:
		std::vector<std::vector<dataType>> data;
		uint32_t length; // Number of rows
		uint16_t col_length; // Number of columns
		uint16_t y_index; // Index of the training output
		std::vector<Column_Summary> column_summary; // holds summary of each column

		void handle_null_values();
		void summarize_columns();
		int16_t get_col_index(std::string name);
};

#endif // DATASET_H
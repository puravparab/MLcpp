#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>
#include <fstream>
#include "../../src/dataset/dataset.h"
#include "../../src/utilities/type.cpp"

using dataType = std::variant<std::string, float>;

Column_Summary::Column_Summary(std::string name) : name(name), type(std::type_index(typeid(float))){}
Dataset::Dataset(){}

// Read the dataset from the file path (csv only for now)
void Dataset::read(const std::string file_path){
  std::ifstream file(file_path);
  if (!file.is_open()) {
  	std::cerr << "Error: Could not open file: " << file_path << std::endl;
  	return;
  }
	// Read headers
	std::string line;
	std::getline(file, line);
	std::istringstream header_stream(line);
	std::string header;
	while (std::getline(header_stream, header, ',')) {
		Column_Summary col_summary(header);
		column_summary.push_back(col_summary);
	}
	
	// Determine the number of columns
	col_length = column_summary.size();

	// Read data
	std::vector<std::vector<dataType>> tempData;  // Temporary storage for data
	while (std::getline(file, line)) {
		std::istringstream row_stream(line);
		std::vector<dataType> row;
		std::string value;
		while (std::getline(row_stream, value, ',')) {
			row.push_back(value);
		}
		tempData.push_back(row);
	}
	length = tempData.size();
	data = tempData;

	handle_null_values();
	summarize_columns();
}

// Handle null values
void Dataset::handle_null_values() {
	for (uint16_t i = 0; i < col_length; i++){
		uint32_t num_strings = 0;
		uint32_t num_float = 0;
		for (uint16_t j = 0; j < length; j++){
			std::string value = std::get<std::string>(data[j][i]);
			// If value is double or int convert to float
			if (is_type_int(value) || is_type_double(value)){
				data[j][i] = stof(value);
				num_float += 1;
			}
			// If value is empty add index to null_index
			else if (value == ""){
				column_summary[i].null_index.push_back(j);
			} else {
				num_strings += 1;
			}
		}

		// very crude solution to determine column type:
		// If column has more floats than strings
		if (num_float > num_strings){
			column_summary[i].type = std::type_index(typeid(float));
			for (uint16_t j = 0; j < column_summary[i].null_index.size(); j++){
				data[column_summary[i].null_index[j]][i] = std::numeric_limits<float>::infinity();
			}
		}
		// If column has more strings than floats
		else {
			for (uint16_t j = 0; j < column_summary[i].null_index.size(); j++){
				data[column_summary[i].null_index[j]][i] = "Null";
			}
			column_summary[i].type = std::type_index(typeid(std::string));
		}
	}
}

// Summarize columns
void Dataset::summarize_columns() {
	for (uint16_t i = 0; i < column_summary.size(); i++){
		// If column has strings
		if (column_summary[i].type == std::type_index(typeid(std::string))){
			for (uint32_t j = 0; j < length; j++){
				// Populate strings map
				std::string value = std::get<std::string>(data[j][i]);
				std::unordered_map<std::string, int>::const_iterator got = column_summary[i].unique_strings.find(value);
				if (got == column_summary[i].unique_strings.end()){
					column_summary[i].unique_strings.insert(std::pair<std::string,int>(value, 1));
				} else {
					column_summary[i].unique_strings.at(value) += 1;
				}
			}
		}
		// If column has floats
		else if (column_summary[i].type == std::type_index(typeid(float))){
			column_summary[i].sum = 0;
			column_summary[i].max = std::numeric_limits<float>::infinity() * -1;
			column_summary[i].min = std::numeric_limits<float>::infinity();
			uint32_t null_count = column_summary[i].null_index.size();

			// Calculate mean, max, min, sum
			for (uint32_t j = 0; j < length; j++){
				float value = std::get<float>(data[j][i]);
				if (value != std::numeric_limits<float>::infinity()){
					column_summary[i].sum += value;
					if (value > column_summary[i].max){column_summary[i].max = value;}
					if (value < column_summary[i].min){column_summary[i].min = value;}
				}
			}
			column_summary[i].mean = column_summary[i].sum / (length - null_count);

			// Calculate standard deviation
			float rms_sum = 0;
			for (uint32_t j = 0; j < length; j++){
				float value = std::get<float>(data[j][i]);
				if (value != std::numeric_limits<float>::infinity()){
					rms_sum += pow((value - column_summary[i].mean), 2);
				}
			}
			column_summary[i].std_dev = sqrt(rms_sum / (length - null_count));
		}
	}
}

// Return shape: (rows, columns)
const std::vector<uint32_t> Dataset::shape(){
	return {length, col_length};
}

// Print first n rows
const void Dataset::head(uint8_t n, int width){
	printf("\nHEAD: First %d elements\n", n);
	for (uint16_t i = 0; i < col_length; i++){
		std::cout << std::setw(width) << column_summary[i].name << " ";
	}
	printf("\n");
	for (uint16_t i = 0; i < n; i++){
		for (uint16_t j = 0; j < col_length; j++){
			try{
				std::cout << std::setw(width) << std::get<std::string>(data[i][j]) << " ";
			} catch (const std::bad_variant_access&){
				std::cout << std::setw(width) << std::get<float>(data[i][j]) << " ";
			}
		}
		printf("\n");
	}
}

// Print headers
 const void Dataset::print_headers(){
	std::unordered_map<std::type_index, std::string> type_names;
	type_names[std::type_index(typeid(std::string))] = "string";
	type_names[std::type_index(typeid(float))] = "float";
	printf("\nHEADERS:\n");
	for (uint16_t i = 0; i < col_length; i++){
		std::cout << i + 1 << ". " << column_summary[i].name << "(" << type_names[column_summary[i].type] << ")" << std::endl;
	} 
 }

// Display column summary
const void Dataset::col_summary(std::string name){
	int16_t index = -1;
	for (int16_t i = 0; i < col_length; i++){
		if(column_summary[i].name == name){
			index = i;
			break;
		}
	}
	if (index == -1){
		std::cout << "\nError: column " << name << " does not exist" << std::endl;
		return;
	}
	printf("\nCOL SUMMARY:\n");
	printf("Name: %s\n", name.c_str());
	// Column has strings
	if (column_summary[index].type == std::type_index(typeid(std::string))){
		std::cout << "Data: " << column_summary[index].unique_strings.bucket_count() << " unique elements" << std::endl;
		for (auto it = column_summary[index].unique_strings.begin(); it != column_summary[index].unique_strings.end(); ++it ){
			std::cout << "- " << it->first << ": " << it->second << std::endl;
		}
	}
	// Column has floats
	else if (column_summary[index].type == std::type_index(typeid(float))){
		std::cout << "Mean: " << column_summary[index].mean << std::endl;
		std::cout << "Max: " << column_summary[index].max << std::endl;
		std::cout << "Min: " << column_summary[index].min << std::endl;
		std::cout << "Std Dev: " << column_summary[index].std_dev << std::endl;
		std::cout << "Null values: " << column_summary[index].null_index.size() << std::endl;
	}
}
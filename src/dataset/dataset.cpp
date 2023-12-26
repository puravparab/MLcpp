#include <iostream>
#include <fstream>
#include "dataset.h"

Header_Item::Header_Item(std::string name, std::type_index type) : name(name), type(type){}
Dataset::Dataset(){}

// Read the dataset from the file path (csv only for now)
void Dataset::read(const std::string& file_path){
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
		Header_Item header_item(header, std::type_index(typeid(float)));
		headers.push_back(header_item);
	}
	
	// Determine the number of columns
	col_length = headers.size();

	// Read data into Eigen matrix
	std::vector<float> row;
	while (std::getline(file, line)) {
		std::istringstream row_stream(line);
		float value;
		while (row_stream >> value) {
			row.push_back(value);
			if (row_stream.peek() == ',') {
				row_stream.ignore();
			}
		}
	}

	// Determine the number of rows
	length = row.size() / col_length;

	// Populate Eigen matrix 
	data = Eigen::Map<Eigen::MatrixXf>(row.data(), length, col_length);
}

// Return shape
const std::vector<uint32_t> Dataset::shape(){
	return {length, col_length};
}

// Print first n columns
const void Dataset::head(uint8_t n){
	for (uint16_t i = 0; i < col_length; i++){
		std::cout << headers[i].name << " " << std::endl;
	}
	printf("\n");
	for (uint16_t i = 0; i < n; i++){
		for (uint16_t j = 0; j < col_length; j++){
			std::cout << data(i ,j) << " " << std::endl;
		}
		printf("\n");
	}
}
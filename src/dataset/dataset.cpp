#include <iostream>
#include <iomanip>
#include <fstream>
#include "../../src/dataset/dataset.h"
#include "../../src/utilities/type.cpp"

using dataType = std::variant<std::string, float>;

Header_Item::Header_Item(std::string name, std::type_index type) : name(name), type(type){}
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
		Header_Item header_item(header, std::type_index(typeid(std::string)));
		headers.push_back(header_item);
	}
	
	// Determine the number of columns
	col_length = headers.size();

	// Read data
	std::vector<std::vector<dataType>> tempData;  // Temporary storage for data
	while (std::getline(file, line)) {
		std::istringstream row_stream(line);
		std::vector<dataType> row;
		std::string value;
		while (std::getline(row_stream, value, ',')) {
			if (is_type_int(value)){
				row.push_back(stof(value));
			} else if (is_type_double(value)){
				row.push_back(stof(value));
			} else{
				row.push_back(value);
			}
		}
		tempData.push_back(row);
	}
	length = tempData.size();
	data = tempData;

	update_header_type();
}

// Update header type
void Dataset::update_header_type(){
	// Update header type
	for (uint16_t i = 0; i < col_length; i++) {
		std::type_index temp = std::type_index(typeid(std::string));
		// Check if float 
		if (!data.empty() && std::holds_alternative<float>(data[0][i])){
			temp = std::type_index(typeid(float));
		}
		headers[i].type = temp;
	}
}

// Return shape
const std::vector<uint32_t> Dataset::shape(){
	return {length, col_length};
}

// Print first n rows
const void Dataset::head(uint8_t n){
	for (uint16_t i = 0; i < col_length; i++){
		std::cout << std::setw(16) << headers[i].name << " ";
	}
	printf("\n");
	for (uint16_t i = 0; i < n; i++){
		for (uint16_t j = 0; j < col_length; j++){
			try{
				std::cout << std::setw(16) << std::get<std::string>(data[i][j]) << " ";
			} catch (const std::bad_variant_access&){
				std::cout << std::setw(16) << std::get<float>(data[i][j]) << " ";
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
	for (uint16_t i = 0; i < col_length; i++){
		std::cout << i + 1 << ". " << headers[i].name << "(" << type_names[headers[i].type] << ")" << std::endl;
	} 
 }

// Column Summary
const void Dataset::col_summary(std::string name){
	int16_t index = -1;
	for (int16_t i = 0; i < col_length; i++){
		if(headers[i].name == name){
			index = i;
			break;
		}
	}
	if (index == -1){
		std::cout << "Error: column " << name << " does not exist" << std::endl;
		return;
	}
	std::cout << name << std::endl;
}
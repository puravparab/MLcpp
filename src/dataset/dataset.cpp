#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>
#include <fstream>
#include <unordered_map>
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
			// If type is int convert to float
			if (is_type_int(value)){
				row.push_back(stof(value));
			} 
			// If type is double convert to float
			else if (is_type_double(value)){
				row.push_back(stof(value));
			} 
			// If type is string don't do anything
			else{
				if (value == ""){value = "";}
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
const void Dataset::head(uint8_t n, int width){
	printf("\nHEAD: First %d elements\n", n);
	for (uint16_t i = 0; i < col_length; i++){
		std::cout << std::setw(width) << headers[i].name << " ";
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
		std::cout << "\nError: column " << name << " does not exist" << std::endl;
		return;
	}
	printf("\nCOL SUMMARY:\n");
	printf("Name: %s\n", name.c_str());

	// Column has strings
	if (headers[index].type == std::type_index(typeid(std::string))){
		std::unordered_map<std::string, int> strings; // Map that tracks unique strings
		for (uint32_t i = 0; i < length; i++){
			// Populate strings map
			std::string value = std::get<std::string>(data[i][index]);
			std::unordered_map<std::string, int>::const_iterator got = strings.find(value);
			if (got == strings.end()){
				if (value == ""){value = "Null";}
				strings.insert(std::pair<std::string,int>(value,1));
			} else {
				strings.at(value) += 1;
			}
		}

		std::cout << "Data: " << strings.bucket_count() << " unique elements" << std::endl;
		for (auto it = strings.begin(); it != strings.end(); ++it ){
			std::cout << "- " << it->first << ": " << it->second << std::endl;
		}
	}

	// Column has floats
	else if (headers[index].type == std::type_index(typeid(float))){
		float sum = 0;
		float max = std::numeric_limits<float>::infinity() * -1;
		float min = std::numeric_limits<float>::infinity();
		uint32_t null_count = 0;
		for (uint32_t i = 0; i < length; i++){
			float value = std::get<float>(data[i][index]);
			sum += value;
			if (value > max){max = value;}
			if (value < min){min = value;}
		}
		float mean = sum / length;

		float rms_sum = 0;
		for (uint32_t i = 0; i < length; i++){
			float value = std::get<float>(data[i][index]);
			rms_sum += pow((value - mean), 2);
		}
		float std_dev = sqrt(rms_sum / length);
		std::cout << "Mean: " << mean << std::endl;
		std::cout << "Max: " << max << std::endl;
		std::cout << "Min: " << min << std::endl;
		std::cout << "Std Dev: " << std_dev << std::endl;
	}
}
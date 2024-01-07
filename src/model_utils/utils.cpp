#include <iostream>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "utils.h"

using json = nlohmann::json;

void save_weights(Eigen::VectorXf& model_weights, int param_count, std::string save_type, std::string filename){
  json data;
  std::vector<float> weights;
  for (int i = 0; i < param_count; i++){
    weights.push_back(model_weights(i));
  }
  
  data["weights"] = weights;
  data["bias"] = model_weights(param_count);
	
	std::string filepath = "../models/" + filename;
  if (save_type == "json"){
		std::filesystem::create_directory("../models");
		std::ofstream outFile(filepath);
		if (outFile.is_open()) {
			outFile << data.dump(2); // Dump data with indentation
			outFile.close();
			std::cout << "Data saved to " << filepath << std::endl;
		} else {
			std::cerr << "Error: Unable to open the file" << std::endl;
			exit(1);
		}
	} else {
		std::cerr << "Invalid save_type. Supported types: 'json'" << std::endl;
  }
}

Eigen::VectorXf load_weights(const std::string& filename) {
	std::ifstream inFile(filename);
	if (!inFile.is_open()) {
		std::cerr << "Error: Unable to open the file" << std::endl;
		exit(1);
	}
	json data;
	inFile >> data;
	
	Eigen::VectorXf model_weights(data["weights"].size() + 1);
	// Load weights
	for (size_t i = 0; i < data["weights"].size(); ++i) {
		model_weights(i) = data["weights"][i];
	}
	// Load bias
	model_weights(data["weights"].size()) = data["bias"];
	inFile.close();
	return model_weights;
}
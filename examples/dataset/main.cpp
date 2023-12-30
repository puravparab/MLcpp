#include <iostream>
#include <vector>
#include "../../src/dataset/dataset.h"

// Run ./setup.sh -> enter "dataset"

int main() {
  Dataset dataset;
	// read dataset
	dataset.read("../../../datasets/housing.csv");

	// set output column
	dataset.set_output_column("median_house_value");
	
	// display shape
	const std::vector<uint32_t> shape = dataset.shape();
	std::cout << "Shape: (" << shape[0] << ", " << shape[1] <<  ")" << std::endl;

	// print headers
	dataset.print_headers();

	// print first three rows
	dataset.head(3, 15);

	// display summary for every column
	std::vector<std::string> headers = dataset.get_headers();
	for (uint32_t i = 0; i < headers.size(); i++){
		dataset.col_summary(headers[i]);
	}
}

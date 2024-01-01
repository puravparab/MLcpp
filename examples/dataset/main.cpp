#include <iostream>
#include <vector>

#include "../../src/dataset/dataset.h"
#include "../../src/dataset/dataloader.h"

// Run ./setup.sh -> enter "dataset"
int main() {
  Dataset dataset;
	// read dataset
	dataset.read("../../../datasets/housing.csv");

	// set output column
	dataset.set_output_column("median_house_value");

	// display shape
	std::vector<uint32_t> shape = dataset.shape();
	std::cout << "Shape: (" << shape[0] << ", " << shape[1] <<  ")" << std::endl;

	// print headers
	dataset.print_headers();

	// print first three rows
	dataset.head(3, 15);

	// drop null rows
	dataset.drop_null_rows();

	// create one hot encoding for ocean_proximity
	dataset.one_hot_encoding("ocean_proximity");
	dataset.print_headers();

	// display summary for every column
	std::vector<std::string> headers = dataset.get_headers();
	for (uint32_t i = 0; i < headers.size(); i++){
		dataset.col_summary(headers[i]);
	}

	// display updated shape
	shape = dataset.shape();
	std::cout << "\nShape: (" << shape[0] << ", " << shape[1] <<  ")" << std::endl;
}

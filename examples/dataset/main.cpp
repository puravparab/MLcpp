#include <iostream>
#include <vector>
#include "../../src/dataset/dataset.h"

int main() {
  Dataset dataset;
	// read dataset
	dataset.read("../../../datasets/housing.csv");

	// display shape
	const std::vector<uint32_t> shape = dataset.shape();
	std::cout << "Shape: (" << shape[0] << ", " << shape[1] <<  ")" << std::endl;

	// print headers
	dataset.print_headers();

	// print first three rows
	dataset.head(3, 15);

	// display column summary
	dataset.col_summary("housing_median_age");
	dataset.col_summary("longitude");
	dataset.col_summary("latitude");
	dataset.col_summary("total_rooms");
	// dataset.col_summary("total_bedrooms");
	dataset.col_summary("population");
	dataset.col_summary("households");
	dataset.col_summary("median_income");
	dataset.col_summary("median_house_value");
	dataset.col_summary("ocean_proximity");
}

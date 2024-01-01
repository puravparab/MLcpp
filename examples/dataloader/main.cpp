#include <iostream>
#include <vector>

#include "../../src/dataset/dataset.h"
#include "../../src/dataset/dataloader.h"

// Run ./setup.sh -> enter "dataloader"
int main() {
  Dataset dataset;
	// read dataset
	dataset.read("../../../datasets/housing.csv");

	// set output column
	dataset.set_output_column("median_house_value");

	// drop null rows
	dataset.drop_null_rows();

	// create one hot encoding for ocean_proximity
	dataset.one_hot_encoding("ocean_proximity");
	dataset.print_headers();
 
	// display updated shape
	std::vector<uint32_t> shape = dataset.shape();
	std::cout << "\nShape: (" << shape[0] << ", " << shape[1] <<  ")\n" << std::endl;

	auto d = dataset.get_data();
	Dataloader dl(d);
	std::vector<Eigen::MatrixXf> vec = dl.split(90, 10);
	Eigen::MatrixXf subset = vec[0].block(0, 0, 10, 14);
	std::cout << "Eigen Matrix:\n" << subset.format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]")) << std::endl;
}

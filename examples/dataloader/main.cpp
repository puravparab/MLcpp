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

	Dataloader dl(dataset);
	auto dl_split = dl.split(90, 10);
  auto x_train = dl_split[0][0];
  auto y_train = dl_split[0][1];
  auto x_test = dl_split[1][0];
  auto y_test = dl_split[1][1];
	Eigen::MatrixXf subset = x_train.block(0, 0, 10, 13);
	std::cout << "X Train:\n" << subset.format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]")) << std::endl;
}

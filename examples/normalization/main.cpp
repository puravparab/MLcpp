#include <iostream>
#include "../../src/dataset/dataset.h"
#include "../../src/dataset/dataloader.h"
#include "../../src/normalization/norm.h"

int main(){
  Dataset dataset;
	// read dataset
	dataset.read("../../../datasets/housing.csv");
	// set output column
	dataset.set_output_column("median_house_value");
	// drop null rows
	dataset.drop_null_rows();
	// create one hot encoding for ocean_proximity
	dataset.one_hot_encoding("ocean_proximity");
 
	// display updated shape
	std::vector<uint32_t> shape = dataset.shape();
	std::cout << "\nShape: (" << shape[0] << ", " << shape[1] <<  ")\n" << std::endl;

  // create dataloader
	Dataloader dl(dataset);
	auto dl_split = dl.split(90, 10);
  Eigen::MatrixXf subset = dl_split[0][0].block(0, 0, 5, 13);
	std::cout << "X Train:\n" << subset.format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]")) << "\n" << std::endl;

  // normalize using z_score
  Norm norm("z_score", dl);
  std::vector<Eigen::MatrixXf> norm_vec {dl_split[0][0]};
  norm.normalize(norm_vec);

  // display results
  subset = norm_vec[0].block(0, 0, 5, 13);
	std::cout << "Normalized X Train:\n" << subset.format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]")) << "\n" << std::endl;
}
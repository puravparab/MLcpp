#include <iostream>
#include "../../src/dataset/dataset.h"
#include "../../src/dataset/dataloader.h"
#include "../../src/normalization/norm.h"
#include "../../src/loss/loss.h"

int main(){
  Dataset dataset;
	dataset.read("../../../datasets/housing.csv"); // read dataset
	dataset.set_output_column("median_house_value"); // set output column
	dataset.drop_null_rows(); // drop null rows
  dataset.one_hot_encoding("ocean_proximity"); // create one hot encoding for ocean_proximity

	Dataloader dl(dataset); // create dataloader
	auto dl_split = dl.split(90, 10);
  // normalize
  Norm norm("z_score", dl);
  std::vector<Eigen::MatrixXf> norm_vec {dl_split[0][0], dl_split[1][0]};
  norm.normalize(norm_vec);

 
  auto x_train = norm_vec[0];
  auto y_train = dl_split[0][1] / 100000;

  // create arbitrary weights
  Eigen::MatrixXf w(x_train.cols(), 1);
  w << 1, 2, 4, 5, 4, 2 , 3, 6, 4, 2, 1, 4 , 5;

  // calculate loss
  auto y_predict = x_train * w;
  auto loss = mse(y_predict, y_train);

  // display results
  Eigen::MatrixXf subset = y_predict.block(0, 0, 6, 1);
	std::cout << "Y Predict:\n" << subset.format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]")) << "\n" << std::endl;
  subset = y_train.block(0, 0, 6, 1);
	std::cout << "Y Train:\n" << subset.format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]")) << "\n" << std::endl;
  std::cout << "Loss: " << loss << std::endl;
}
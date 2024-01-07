#include <iostream>
#include "../../src/dataset/dataset.h"
#include "../../src/dataset/dataloader.h"
#include "../../src/normalization/norm.h"
#include "../../src/loss/loss.h"
#include "../../src/optimizers/optim.h"
#include "../../src/model_utils/utils.h"

int main(){
  Dataset dataset;
	dataset.read("../../../datasets/housing.csv"); // read dataset
	dataset.set_output_column("median_house_value"); // set output column
	dataset.drop_column("latitude");
	dataset.drop_column("longitude");
	dataset.drop_column("ocean_proximity");
	dataset.drop_null_rows(); // drop null rows

	Dataloader dl(dataset); // create dataloader
	auto dl_split = dl.split(90, 10);

  // normalize
  Norm norm("z_score", dl);
  std::vector<Eigen::MatrixXf> norm_vec {dl_split[0][0], dl_split[1][0]};
  norm.normalize(norm_vec);

	int scale = 100000; // scale down output values
  Eigen::MatrixXf x_train = norm_vec[0];
  Eigen::MatrixXf y_train = dl_split[0][1] / scale;
  
	// test
	Eigen::MatrixXf x_test = norm_vec[1]; // normalized test inputs
	Eigen::MatrixXf y_test = dl_split[1][1] / scale;

  // load weights and bias
  Eigen::VectorXf weights = load_weights("../models/weights.json");
  Eigen::VectorXf W = weights.segment(0, x_train.cols());
  float B = weights(weights.rows() - 1);

  // run predictions
  Eigen::VectorXf pred_1 = x_train * W + Eigen::VectorXf::Constant(x_train.rows(), B);
  float loss_1 = mse(pred_1, y_train);
  Eigen::VectorXf pred_2 = x_test * W + Eigen::VectorXf::Constant(x_test.rows(), B);
  float loss_2 = mse(pred_2, y_test);

  printf("\nResults:\n");
  std::cout << "Weights: " << W.transpose().format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]")) << std::endl;
	std::cout << "Bias: " << B << std::endl;
	std::cout << "Test loss 1: " << loss_1 << std::endl;
  std::cout << "Test loss 2: " << loss_2 << std::endl;
}
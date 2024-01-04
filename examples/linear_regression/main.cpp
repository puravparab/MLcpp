#include <iostream>
#include "../../src/dataset/dataset.h"
#include "../../src/dataset/dataloader.h"
#include "../../src/normalization/norm.h"
#include "../../src/loss/loss.h"
#include "../../src/optimizers/optim.h"

void linear_regression(Eigen::MatrixXf&X, Eigen::MatrixXf& Y, Eigen::VectorXf& W, float& B, float lr, int epochs){
  int steps;
  if (epochs > 10){
    steps = epochs / 10;
  } else {
    steps = epochs;
  }
  Eigen::VectorXf history;
  for (int i = 0; i < epochs; i++){
    history = batchgd(X, Y, W, B, lr, mse);
    if (i % steps == 10){
      std::cout << "Epoch: " << i + 1 << " " << "loss: " << history[history.rows() - 1] << std::endl;
    }
    W = history.head(X.cols());
    B = history[X.cols()];
  }
  std::cout << "Epoch: " << epochs << " " << "loss: " << history[history.rows() - 1] << std::endl;
}

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

 
  Eigen::MatrixXf x_train = norm_vec[0];
  Eigen::MatrixXf y_train = dl_split[0][1] / 100000;
  
  // run linear regression
  Eigen::VectorXf W = Eigen::VectorXf::Zero(x_train.cols());
  float B = 1;
  linear_regression(x_train, y_train, W, B, 0.001, 4000);

  // display results
	std::cout << "\nWeights: \n" << W.format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]")) << "\n" << std::endl;
  std::cout << "\nBias: " << B << std::endl;
}
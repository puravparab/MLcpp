#include <iostream>
#include "../../src/dataset/dataset.h"
#include "../../src/dataset/dataloader.h"
#include "../../src/normalization/norm.h"
#include "../../src/loss/loss.h"
#include "../../src/optimizers/optim.h"
#include "../../src/model_utils/utils.h"

// runs linear regression with batch gradient descent
void linear_regression_bgd(Eigen::MatrixXf&X, Eigen::MatrixXf& Y, Eigen::VectorXf& W, float& B, float lr, int epochs){
  int steps;
  if (epochs > 10){
    steps = epochs / 10;
  } else {
    steps = epochs;
  }
  Eigen::VectorXf history;
	printf("\nBatch gradient descent\n");
	printf("Training starting...\n");
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

// runs linear regression with mini batch gradient descent
void linear_regression_mbgd(Eigen::MatrixXf&X, Eigen::MatrixXf& Y, Eigen::VectorXf& W, float& B, float lr, int epochs, int batch){
  Eigen::MatrixXf history;
	printf("\nMini-Batch gradient descent\n");
	printf("Training starting...\n");
  for (int i = 0; i < epochs; i++){
    history = mbgd(X, Y, W, B, lr, mse, batch);
    std::cout << "Epoch: " << i + 1 << " " << "loss: " << history(history.rows() - 1, history.cols() - 1) << std::endl;
    W = history.block(history.rows() - 1, 0, 1, W.rows()).transpose();
    B = history(history.rows() - 1, history.cols() - 2);
  }
}

// runs linear regression with stochastic gradient descent
void linear_regression_sgd(Eigen::MatrixXf&X, Eigen::MatrixXf& Y, Eigen::VectorXf& W, float& B, float lr, int epochs, int batch){
  Eigen::MatrixXf history;
	printf("\nStochastic gradient descent\n");
	printf("Training starting...\n");
  for (int i = 0; i < epochs; i++){
    history = sgd(X, Y, W, B, lr, mse, batch);
    std::cout << "Epoch: " << i + 1 << " " << "loss: " << history(history.rows() - 1, history.cols() - 1) << std::endl;
    W = history.block(history.rows() - 1, 0, 1, W.rows()).transpose();
    B = history(history.rows() - 1, history.cols() - 2);
  }
}

int main(){
  Dataset dataset;
	dataset.read("../../../datasets/housing.csv"); // read dataset
	dataset.set_output_column("median_house_value"); // set output column
	dataset.drop_column("latitude");
	dataset.drop_column("longitude");
	dataset.drop_column("ocean_proximity");
	dataset.drop_null_rows(); // drop null rows

	dataset.print_headers();
	std::vector<uint32_t> shape = dataset.shape();
	printf("\nShape: (%d,%d)\n", shape[0], shape[1]);

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

  // run linear regression
	// batch gradient descent
  float learning_rate = 0.0009;
  int epochs = 5000;
  Eigen::VectorXf W_1 = Eigen::VectorXf::Zero(x_train.cols());
  float B_1 = 1;
  linear_regression_bgd(x_train, y_train, W_1, B_1, learning_rate, epochs);
	Eigen::VectorXf test_pred = x_test * W_1 + Eigen::VectorXf::Constant(x_test.rows(), B_1);
	float test_loss_1 = mse(test_pred, y_test);
	Eigen::VectorXf weights_1(W_1.size() + 1);
	weights_1 << W_1, B_1;
	save_weights(weights_1, W_1.size(), "json", "lr_bgd.json");

	// mini-batch gradient descent
	learning_rate = 0.003;
  epochs = 30;
	int batch = 64;
  Eigen::VectorXf W_2 = Eigen::VectorXf::Zero(x_train.cols());
	float B_2 = 1;
	linear_regression_mbgd(x_train, y_train, W_2, B_2, learning_rate, epochs, batch);
	test_pred = x_test * W_2 + Eigen::VectorXf::Constant(x_test.rows(), B_2);
	float test_loss_2 = mse(test_pred, y_test);
	Eigen::VectorXf weights_2(W_2.size() + 1);
	weights_2 << W_2, B_2;
	save_weights(weights_2, W_2.size(), "json", "lr_mbgd.json");

	// stochastic gradient descent
	learning_rate = 0.0001;
  epochs = 30;
	batch = 64;
  Eigen::VectorXf W_3 = Eigen::VectorXf::Zero(x_train.cols());
	float B_3 = 1;
	linear_regression_sgd(x_train, y_train, W_3, B_3, learning_rate, epochs, batch);
	test_pred = x_test * W_3 + Eigen::VectorXf::Constant(x_test.rows(), B_3);
	float test_loss_3 = mse(test_pred, y_test);
	Eigen::VectorXf weights_3(W_3.size() + 1);
	weights_3 << W_3, B_3;
	save_weights(weights_3, W_3.size(), "json", "lr_sgd.json");

	std::cout<< "\nBatch gradient descent:" << std::endl;
	std::cout << "Weights: " << W_1.transpose().format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]")) << std::endl;
	std::cout << "Bias: " << B_1 << std::endl;
	std::cout << "Test loss: " << test_loss_1 << std::endl;

	std::cout << "\nMini-Batch gradient descent:" << std::endl;
	std::cout << "Weights: " << W_2.transpose().format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]")) << std::endl;
	std::cout << "Bias: " << B_2 << std::endl;
	std::cout << "Test loss: " << test_loss_2 << std::endl;

	std::cout << "\nStochastic gradient descent:" << std::endl;
	std::cout << "Weights: " << W_3.transpose().format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]")) << std::endl;
	std::cout << "Bias: " << B_3 << std::endl;
	std::cout << "Test loss: " << test_loss_3 << std::endl;
}
#include <iostream>
#include "../../src/dataset/dataset.h"
#include "../../src/dataset/dataloader.h"
#include "../../src/normalization/norm.h"
#include "../../src/loss/loss.h"
#include "../../src/optimizers/optim.h"
#include "../../src/model_utils/utils.h"

// calculate sigmoid
Eigen::VectorXf sigmoid(Eigen::VectorXf X){
	return 1.0 / (1.0 + (-X.array()).exp());
}

void logistic_regression(Eigen::MatrixXf&X, Eigen::MatrixXf& Y, Eigen::VectorXf& W, float& B, float lr, int epochs){
	Eigen::VectorXf history;
	printf("\nBatch gradient descent\n");
	printf("Training starting...\n");
	for (int i = 0; i < epochs; i++){
    history = batchgd(X, Y, W, B, lr, bce);
    std::cout << "Epoch: " << i + 1 << " " << "loss: " << history[history.rows() - 1] << std::endl;
    W = history.head(X.cols());
    B = history[X.cols()];
  }
  std::cout << "Epoch: " << epochs << " " << "loss: " << history[history.rows() - 1] << std::endl;
}

int main(){
  Dataset dataset;
	dataset.read("../../../datasets/breast_cancer.csv"); // read dataset
	dataset.set_output_column("Class"); // set output column

  dataset.print_headers(); // print headers
  // print column summary
  std::vector<std::string> headers = dataset.get_headers();
	for (uint32_t i = 0; i < headers.size(); i++){
		dataset.col_summary(headers[i]);
	}
  // print shape
  std::vector<uint32_t> shape = dataset.shape();
	printf("\nShape: (%d,%d)\n", shape[0], shape[1]);
  
  Dataloader dl(dataset); // create dataloader
	auto dl_split = dl.split(90, 10);
  // train
  Eigen::MatrixXf x_train = dl_split[0][0];
  Eigen::MatrixXf y_train = dl_split[0][1];
	for (size_t i = 0; i < y_train.size(); i++){
		if (y_train(i) == 2){y_train(i) = 0;}
		else if(y_train(i) == 4){y_train(i) = 1;}
	}
	
	// test
	Eigen::MatrixXf x_test = dl_split[1][0];
	Eigen::MatrixXf y_test = dl_split[1][1];
	for (size_t i = 0; i < y_test.size(); i++){
		if (y_test(i) == 2){y_test(i) = 0;}
		else if(y_test(i) == 4){y_test(i) = 1;}
	}

	// run logistic regression
	// batch gradient descent
  float learning_rate = 0.001;
  int epochs = 80;
  Eigen::VectorXf W_1 = Eigen::VectorXf::Zero(x_train.cols());
  float B_1 = 1;
  logistic_regression(x_train, y_train, W_1, B_1, learning_rate, epochs);
	Eigen::VectorXf test_pred = sigmoid(x_test * W_1 + Eigen::VectorXf::Constant(x_test.rows(), B_1));
	float test_loss_1 = bce(test_pred, y_test);
	Eigen::VectorXf weights_1(W_1.size() + 1);
	weights_1 << W_1, B_1;
	save_weights(weights_1, W_1.size(), "json", "weights.json");

	std::cout<< "\nBatch gradient descent:" << std::endl;
	std::cout << "Weights: " << W_1.transpose().format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]")) << std::endl;
	std::cout << "Bias: " << B_1 << std::endl;
	std::cout << "Test loss: " << test_loss_1 << std::endl;
}
#include <iostream>
#include <vector>
#include "../../src/dataset/dataset.h"

int main() {
  Dataset dataset;
	dataset.read("../../../datasets/housing.csv");
	dataset.head(10);
	std::vector shape = dataset.shape();
	std::cout << "Shape: " << shape[0] << " " << shape[1];
}

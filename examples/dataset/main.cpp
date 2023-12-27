#include <iostream>
#include <vector>
#include "../../src/dataset/dataset.h"

int main() {
  Dataset dataset;
	dataset.read("../../../datasets/housing.csv");
	dataset.print_headers();

	printf("\n");
	dataset.head(3);

	dataset.col_summary("longitude");
}

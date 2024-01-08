#include <iostream>
#include "../../src/dataset/dataset.h"
#include "../../src/dataset/dataloader.h"
#include "../../src/normalization/norm.h"
#include "../../src/loss/loss.h"
#include "../../src/optimizers/optim.h"

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
  
}
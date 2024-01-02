#include <iostream>
#include <vector>
#include "norm.h"

Norm::Norm(std::string norm_type, Dataloader& dl) : norm_type(norm_type), dl(dl), column_summary(dl.column_summary){
  // validate norm_type
  for (auto i = 0; i < norm_options.size(); i++){
    if (norm_type == norm_options[i]){break;}
    if (i == norm_options.size() - 1){
      std::cerr << "Error: " << norm_type << " does not exist as an option." << std::endl;
      exit(1);
    }
  }
}

// Normalize every matrix
std::vector<Eigen::MatrixXf>& Norm::normalize(std::vector<Eigen::MatrixXf>& data){
  // Iterate through data vector
  for (auto i = 0; i < data.size(); i++){
    if (norm_type == "min-max"){minmax_norm(data[i]);} 
    else if (norm_type == "z_score"){zscore_norm(data[i]);}
  }
  return data;
}

// min-max normalization
void Norm::minmax_norm(Eigen::MatrixXf& matrix){
  matrix = (matrix.rowwise() - dl.min.transpose()).array().rowwise() / (dl.max - dl.min).transpose().array();
}

// z-score normalization
void Norm::zscore_norm(Eigen::MatrixXf& matrix){
  Eigen::MatrixXf normalized_matrix = matrix;

  normalized_matrix = (normalized_matrix.rowwise() - dl.mean.transpose()).array().rowwise() / dl.std_dev.transpose().array();

  // Exclude binary columns (columns with only 0 and 1) from normalization
  for (int i = 0; i < matrix.cols(); ++i) {
    if (column_summary[i].onehotencoding) {
      normalized_matrix.col(i) = matrix.col(i);
    }
  }
  matrix = normalized_matrix;
}
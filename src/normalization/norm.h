// reference: https://developers.google.com/machine-learning/data-prep/transform/normalization

#ifndef NORM_H
#define NORM_H

#include <string>
#include "../dataset/dataloader.h"

class Norm{
  public:
    Norm(std::string norm_type, Dataloader& dl);
    std::vector<Eigen::MatrixXf>& normalize(std::vector<Eigen::MatrixXf>& data);
    void minmax_norm(Eigen::MatrixXf& matrix);
    void zscore_norm(Eigen::MatrixXf& matrix);
  private:
    std::string norm_type;
    Dataloader& dl;
    std::vector<Column_Summary>& column_summary;
    std::vector<std::string> norm_options {"min-max", "z_score"};
};

#endif // NORM_H
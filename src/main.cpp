#include <iostream>
#include <Eigen3/Eigen/Dense>

using Eigen::MatrixXd;

int main()
{
	std::cout << "Hello World" << std::endl;

	MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;
}
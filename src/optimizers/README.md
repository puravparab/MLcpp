## Optimizers Module
Helpful tools for gradient descent

See [linear regression example](../../examples/linear_regression/main.cpp)

### optim.h
Import [optim.h](./optim.h) module
``` cpp
#include "src/optimizers/optim.h"
```

### full batch gradient descent
```cpp
// @param X: matrix (examples x params)
// @param Y: vector containing true values
// @param W: vector of weights
// @param B: bias term
// @param lr: learning rate
// @param loss_function: loss_function from "loss/loss.h"
// @return: a vector containing weights, bias and loss
Eigen::VectorXf history = batchgd(X, Y, W, B, lr, loss_function);
```

### mini-batch gradient descent
```cpp
// @param X: matrix (examples x params)
// @param Y: vector containing true values
// @param W: vector of weights
// @param B: bias term
// @param lr: learning rate
// @param loss_function: loss_function from "loss/loss.h"
// @param batch: loss_function from "loss/loss.h"
// @return: a vector containing weights, bias and loss
Eigen::MatrixXf mbgd(const Eigen::MatrixXf& X, const Eigen::VectorXf& Y, Eigen::VectorXf& W, float B, float lr, loss_function loss_function, int batch)
```

### stochastic gradient descent
```cpp
// @param X: matrix (examples x params)
// @param Y: vector containing true values
// @param W: vector of weights
// @param B: bias term
// @param lr: learning rate
// @param loss_function: loss_function from "loss/loss.h"
// @param batch: loss_function from "loss/loss.h"
// @return: a vector containing weights, bias and loss
Eigen::MatrixXf sgd(const Eigen::MatrixXf& X, const Eigen::VectorXf& Y, Eigen::VectorXf& W, float B, float lr, loss_function loss_function, int batch)
```
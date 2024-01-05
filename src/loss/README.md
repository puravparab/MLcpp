## Normalization Module
Helpful tools to calculate loss.

See [loss example](../../examples/loss/main.cpp)

### loss.h
Import [loss.h](./loss.h) module
``` cpp
#include "src/loss/loss.h"
```
Calculate loss
```cpp
Eigen::MatrixXf y_predict // model predictions
Eigen::MatrixXf y_train // training outputs

// mean squared error
float mse_loss = mse(y_predict, y_train);
```
## Model Utilities Module
Helpful tools for handling models

See [linear regression example](../../examples/linear_regression/main.cpp) and [loading models examples](../../examples/loading_models/main.cpp)

### utils.h
Import [utils.h](./utils.h) module
``` cpp
#include "src/model_utils/utils.h"
```

Saving models to models/
```cpp
// save in json format
save_weights(<vector of weights>, <param count>, "json", "weights.json");
```
Loading models from models/
```cpp
Eigen::VectorXf weights = load_weights(<path to weights file>);
// weights contains weights and bias
```
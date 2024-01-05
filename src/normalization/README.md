## Normalization Module
Helpful tools to normalize input datasets.

See [normalization example](../../examples/normalization/main.cpp)

### norm.h
Import [norm.h](./norm.h) module
``` cpp
#include "src/normalization/norm.h"
```
Create a Norm instance
```cpp
// Params: (norm_type, dl)
// norm_type - ("min-max", "z_score")
// dl - dataloader instance
Norm norm(<normtype>, <dl>);
```

Normalize data
```cpp
// Params: (data)
// data - vector of input matrices
norm.normalize(<data>);
```
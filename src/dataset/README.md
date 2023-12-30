## Dataset Module
Helpful tools to load datasets.

See [dataset example](../../examples/dataset/main.cpp) for reference.

### API
Import dataset module [dataset.h](./dataset.h)
``` cpp
#include "src/dataset/dataset.h"
```
Create a dataset instance
```cpp
Dataset dataset;
```
Load dataset
``` cpp
dataset.read(<path to csv file>);
```
Get shape
```cpp
const std::vector<uint32_t> shape = dataset.shape();
```
Get a list of headers
```cpp
std::vector<std::string> headers = dataset.get_headers();
```
Display headers
```cpp
dataset.print_headers();
```
Display head
```cpp
dataset.head(<number of rows>, <display width>);
```
Display summary for a column
```cpp
dataset.col_summary(<name of column>);
```
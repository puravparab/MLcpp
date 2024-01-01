## Dataset Module
Helpful tools to load datasets.

See [dataset example](../../examples/dataset/main.cpp), [dataloader example](../../examples/dataloader/main.cpp) for reference.

### Dataset.h
Import [dataset.h](./dataset.h) module
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
Set output column
```cpp
dataset.set_output_column(<name of the column>)
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
Remove column from dataset
```cpp
dataset.drop_column(<column name>);
```
Create one hot encoding
```cpp
dataset.one_hot_enconding(<column name>);
```

### Dataloader.h
Import [dataloader.h](./dataloader.h) module
``` cpp
#include "src/dataloader/dataloader.h"
```
Create dataloader
```
Dataloader dl(<Dataset instance>);
```
Split into train and test
```cpp
auto dl_split = dl.split(90, 10);
auto x_train = dl_split[0][0];
auto y_train = dl_split[0][1];
auto x_test = dl_split[1][0];
auto y_test = dl_split[1][1];
```
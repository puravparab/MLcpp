<p align="center">
</p>

<p align="center">
	<h1 align="center">
		MLcpp
	</h1>
	<p align="center">
	    Machine Learning library built in c++
	</p
</p>

# Table of Contents

- [Table of Contents](#table-of-contents)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
	- [Modules](#modules)
- [Examples](#examples)
	- [Linear Regression:](#linear-regression)
	- [Logistic Regression:](#logistic-regression)
- [License](#license)


# Requirements

C++ version: 14

cmake version: 3.6.2

---

# Installation

Clone the repository
```
git clone https://github.com/puravparab/MLcpp.git
```
Change working directory to MLcpp
```
cd MLcpp
```
Create Makefile
```
cmake .
```
Build executable
```
make
```
Run program
```
.\MLcpp.exe
```

---

# Usage

## Modules

-	Dataset:
	- Import datasets and split into training and test data using Dataset class

- Loss
	- Contains classes for various loss functions

- Optimizers
	- Contains classes for various optimizers

- Preprocessing
	- Normalization class for input features

- Regression
	- Contains clases for various linear and logistic regression

---

# Examples

## Linear Regression:

Ex: Create a linear regression model to predict home prices using the real_estate.csv dataset.

In src/main.cpp:

Import dataset and training examples
```
std::string url = ".\\dataset\\real_estate.csv";
Dataset data(url, 80);
MatrixXd train = dataset.get_train();
MatrixXd test = dataset.get_test();
```
Split into features and targets
```
MatrixXd x_train = train.block(0, 0, train.rows(), train.cols() - 1);
MatrixXd y_train = train.col(train.cols() - 1);
MatrixXd x_test = test.block(0, 0, test.rows(), test.cols()-1);
MatrixXd y_test = test.col(test.cols()-1);
```
Scale targets
```
int scale = 1000000;
y_train = y_train / scale;
y_test = y_test / scale;
```
Normalize input features:
```
Normalization normalized(x_train);
x_train = normalized.get_x_train();
x_test = normalized.process(x_test);
```
Add initial weights, bias, learning rate, epsilon and iterations
```
MatrixXd weights{
    {0},{0},{0},{0}
};
double bias = 0;
double learning_rate = 2e-2;
double epsilon = 1e-5;
double iterations = 100000;
```
Create the linear model with batch gradient descent:
```
Linear linear(x_train, y_train, weights, bias);
```
Train linear model:
```
MatrixXd y_predict = linear.train(x_test, y_test, learning_rate, "bgd", epsilon, iterations, 1000);
```
Evaluate trained model with test data
```
x_test = normalized.process(x_test);
std::cout << "Test error: " << linear.evaluate(x_test, y_test) << std::endl;
```
Predict with different values:
```
// x1: bedrooms
// x2: bathrooms
// x3: size of home (sqft)
// x4: size of lot (sqft)
// y = price of home (dollars)
MatrixXd x{
  {2, 4, 2400, 3000},
	{4, 6, 2800, 3200}
};
x = normalized.process(x);
std::cout << "Predictions:\n" << linear.predict(x) * scale << " dollars" << std::endl;
```

## Logistic Regression:

Ex: Create a logistic regression model to perform binary classification using the binary_test1.csv dataset

In src/main.cpp:

Import dataset and training examples
```
std::string url = ".\\dataset\\binary_test1.csv";
Dataset data(url);
MatrixXd train = data.get_train();
```
Split into features and targets
```
MatrixXd x_train = train.block(0, 0, train.rows(), train.cols() - 1);
MatrixXd y_train = train.col(train.cols() - 1);
MatrixXd x_test = test.block(0, 0, test.rows(), test.cols()-1);
MatrixXd y_test = test.col(test.cols()-1);
```
Add initial weights, bias, learning rate, epsilon and iterations
```
MatrixXd weights{
	{0},{0}
};
double bias = 0;
double learning_rate = 0.3;
double epsilon = 1e-6;
double iterations = 5000;
```
Create the logistic model with batch gradient descent:
```
Logistic logistic(x_train, y_train, weights, bias);
```
Train logistic model:
```
MatrixXd y_predict = logistic.train(x_test, y_test, learning_rate, "bgd", epsilon, iterations, 200);
```
Evaluate model with test data:
```
std::cout << "\nTest Evaluation Loss:\n" << logistic.evaluate(x_test, y_test) << std::endl;
```
Predict with different values:
```
std::cout << "\nPredictions: \n" << y_predict << std::endl;

// or 

MatrixXd x{
	{2.5, 3},
	{0.5, 0.5}
};
std::cout << "\nPredictions:\n" << logistic.predict(x) << std::endl;
```

---

# License

MIT License

Copyright (c) 2023 Purav Parab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

Original Creator - [Purav Parab](https://github.com/puravparab)
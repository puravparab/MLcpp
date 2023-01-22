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

<p align="center">
    <a href="#Requirements">Requirements</a>
	&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
	<a href="#Installation">Installation</a>
	&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
	<a href="#Usage">Usage</a>
	&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
	<a href="#License">License</a>
</p>

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

## Linear Regression:

Ex: Create a linear regression model to predict home prices using the real_estate.csv dataset.

In src/main.cpp:

Import dataset and training examples
```
std::string url = ".\\dataset\\real_estate.csv";
Dataset data(url);
MatrixXd x_train = data.get_x_train();
MatrixXd y_train = data.get_y_train();
```
Normalize input:
```
Normalization normalized(x_train);
x_train = normalized.get_x_train();
```
Add initial weights, bias and learning rate:
```
MatrixXd weights{
    {0},{0},{0},{0}
};
double bias = 0;
double learning_rate = 0.01;
```
Create the linear model with stochastic gradient descent:
```
Linear linear(x_train, y_train, weights, bias);
```
Train linear model:
```
MatrixXd y_predict = linear.train(learning_rate, "sgd");
```
Predict with different values:
```
// x1: bedrooms = 5
// x2: bathrooms = 3
// x3: size of home (sqft) = 2400
// x4: size of lot (sqft) = 3000
// y = price of home (dollars)
MatrixXd x{
    {5, 3, 2400, 3000}
};
x = normalized.process(x);
std::cout << "Prediction::\n" << linear.predict(x) << " dollars" << std::endl;
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
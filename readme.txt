Logistic Regression Implementation with Gradient Descent
This project implements logistic regression using batch gradient descent and compares it with stochastic and mini-batch gradient descent methods. It analyzes the performance of each method in terms of training time and accuracy on both training and test datasets.

Overview
The logistic_regression.py script included in this repository implements logistic regression using different gradient descent methods:
Batch Gradient Descent: Iteratively updates model parameters using the entire training dataset.
Stochastic Gradient Descent (SGD): Updates model parameters using a single randomly selected training sample at each iteration.
Mini-Batch Gradient Descent: Updates model parameters using a small random subset of the training dataset at each iteration.

Usage
Clone the repository to your local machine:
git clone https://github.com/yashc1712/Logistic-Regression-with-Gradient-Descent.git

Ensure you have the necessary dependencies installed:
pip install numpy matplotlib

Run the script:
python logistic_regression.py
Check the output in the terminal, which includes training time and accuracy results for each gradient descent method.

Files
Project3_train.csv: Training dataset containing features and labels.
Project3_test.csv: Test dataset containing features and labels.
logistic_regression.py: Python script implementing logistic regression with gradient descent methods.

Results
The script will print the following results:
Training time and accuracy for Stochastic Gradient Descent.
Training time and accuracy for Mini-Batch Gradient Descent.
Training time and accuracy for Batch Gradient Descent.
These results provide insights into the performance of each gradient descent method in terms of both computational efficiency and predictive accuracy.


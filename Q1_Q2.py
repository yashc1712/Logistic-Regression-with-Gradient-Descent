import numpy as np
import math
import time
import matplotlib.pyplot as plt

# Q1: Implement Logistic Regression using Batch Gradient Descent
# Load the training data from Project3_train.csv
data = np.genfromtxt('Project3_train.csv', delimiter=',')
X = data[:, :-1]  # Features
y = data[:, -1]   # Target

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize model parameters
theta = np.zeros(X.shape[1])
learning_rates = [0.001, 0.002, 0.006, 0.01, 0.1]
iterations = 5000
losses = []

# Batch Gradient Descent
for lr in learning_rates:
    theta = np.zeros(X.shape[1])
    loss_history = []
    
    for iteration in range(iterations):
        # Calculate predictions
        predictions = sigmoid(np.dot(X, theta))
        
        # Calculate the gradient and update theta
        gradient = np.dot(X.T, (predictions - y)) / len(y)
        theta -= lr * gradient
        
        # Calculate the loss (log-likelihood)
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        loss_history.append(loss)
    
    losses.append(loss_history)
    
# Plot the loss vs. iteration for different learning rates
plt.figure(figsize=(10, 6))
for i in range(len(learning_rates)):
    plt.plot(range(iterations), losses[i], label=f'LR={learning_rates[i]}')
    
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.title('Logistic Regression Loss vs. Iterations')
plt.savefig('logistic_regression_loss.png')
plt.show()

# Q2: Implement Logistic Regression with Stochastic, Mini-Batch, and Batch Gradient Descent
# Load the training and test data from Project3_train.csv and Project3_test.csv
train_data = np.genfromtxt('Project3_train.csv', delimiter=',')
test_data = np.genfromtxt('Project3_test.csv', delimiter=',')
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize model parameters
theta = np.zeros(X_train.shape[1])
learning_rate = 0.1
iterations = 300000
batch_size = 5

# Helper function for accuracy calculation
def calculate_accuracy(predictions, labels):
    return np.mean((predictions >= 0.5) == labels)

# Stochastic Gradient Descent
theta_sgd = np.zeros(X_train.shape[1])
start_time = time.time()
for iteration in range(iterations):
    random_index = np.random.randint(0, len(X_train))
    x_i = X_train[random_index, :]
    y_i = y_train[random_index]
    
    prediction = sigmoid(np.dot(x_i, theta_sgd))
    gradient = x_i * (prediction - y_i)
    theta_sgd -= learning_rate * gradient

sgd_time = time.time() - start_time

# Mini-Batch Gradient Descent
theta_mini_batch = np.zeros(X_train.shape[1])
start_time = time.time()
for iteration in range(iterations):
    random_indices = np.random.choice(len(X_train), batch_size, replace=False)
    x_batch = X_train[random_indices, :]
    y_batch = y_train[random_indices]
    
    predictions = sigmoid(np.dot(x_batch, theta_mini_batch))
    gradient = np.dot(x_batch.T, (predictions - y_batch)) / batch_size
    theta_mini_batch -= learning_rate * gradient

mini_batch_time = time.time() - start_time

# Batch Gradient Descent
theta_batch = np.zeros(X_train.shape[1])
start_time = time.time()
for iteration in range(iterations):
    predictions = sigmoid(np.dot(X_train, theta_batch))
    gradient = np.dot(X_train.T, (predictions - y_train)) / len(y_train)
    theta_batch -= learning_rate * gradient

batch_time = time.time() - start_time

# Calculate accuracy on training and test data
train_predictions_sgd = sigmoid(np.dot(X_train, theta_sgd))
train_accuracy_sgd = calculate_accuracy(train_predictions_sgd, y_train)
test_predictions_sgd = sigmoid(np.dot(X_test, theta_sgd))
test_accuracy_sgd = calculate_accuracy(test_predictions_sgd, y_test)

train_predictions_mini_batch = sigmoid(np.dot(X_train, theta_mini_batch))
train_accuracy_mini_batch = calculate_accuracy(train_predictions_mini_batch, y_train)
test_predictions_mini_batch = sigmoid(np.dot(X_test, theta_mini_batch))
test_accuracy_mini_batch = calculate_accuracy(test_predictions_mini_batch, y_test)

train_predictions_batch = sigmoid(np.dot(X_train, theta_batch))
train_accuracy_batch = calculate_accuracy(train_predictions_batch, y_train)
test_predictions_batch = sigmoid(np.dot(X_test, theta_batch))
test_accuracy_batch = calculate_accuracy(test_predictions_batch, y_test)

# Print results
print("Stochastic Gradient Descent:")
print("Time taken:", sgd_time, "seconds")
print("Training Accuracy:", train_accuracy_sgd)
print("Test Accuracy:", test_accuracy_sgd)
print()

print("Mini-Batch Gradient Descent:")
print("Time taken:", mini_batch_time, "seconds")
print("Training Accuracy:", train_accuracy_mini_batch)
print("Test Accuracy:", test_accuracy_mini_batch)
print()

print("Batch Gradient Descent:")
print("Time taken:", batch_time, "seconds")
print("Training Accuracy:", train_accuracy_batch)
print("Test Accuracy:", test_accuracy_batch)
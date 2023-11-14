'''
# Method 2
**Steps**
1. Compute the mean of each seen class.
2. Learn the multi-output regression model with class attribute vector being the input and the class mean vector being the output (this will use the seen class attributes and their mean vectors).
3. Apply the learned regression model to compute the mean of each unseen class.
4. Apply the model to predict labels on unseen class test inputs.
5. Compute classification accuracies.

Note: You need to try several values of the regularization hyperparameter $\lambda$
'''

# Importing required libraries
import numpy as np
import pickle as p
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import sys
import os

# Loading the data from respective files
assert len(sys.argv) <= 2
DIR = sys.argv[1]

X_seen=np.load(os.path.join(DIR, 'X_seen.npy'), allow_pickle=True, encoding='bytes')

Xtest=np.load(os.path.join(DIR, 'Xtest.npy'), allow_pickle=True, encoding='bytes')
Ytest=np.load(os.path.join(DIR, 'Ytest.npy'), allow_pickle=True, encoding='bytes')

class_attributes_seen=np.load(os.path.join(DIR, 'class_attributes_seen.npy'), allow_pickle=True, encoding='bytes')
class_attributes_unseen=np.load(os.path.join(DIR, 'class_attributes_unseen.npy'), allow_pickle=True, encoding='bytes')

# Printing the shapes of all the data that is loaded
print(f'''
X_seen:                     {X_seen.shape}
Xtest:                      {Xtest.shape}
Ytest:                      {Ytest.shape}
class_attributes_seen:      {class_attributes_seen.shape}
class_attributes_unseen:    {class_attributes_unseen.shape}

''')

# Computing the means of the seen classes
means_seen = np.array([row.mean(axis=0) for row in X_seen])
print(f'Shape of means: {means_seen.shape}')

def predict(means, test_input):
    distances = np.array([(test_input - mean).T@(test_input - mean) for mean in means])
    # Adding 1 to the predicted value as the range of Ytest is [1, 10]
    # while the below gives a range of [0, 9]
    return np.argmin(distances)+1.0

# `accuracies` notes all the accuracy values for each lambda value
accuracies = []

# Renaming few variables to make typing easy and improve code readability
X = class_attributes_seen
y = means_seen

# all the values of lamda for which we do the below calculations
lambdas = [0.01, 0.1] + list(np.arange(0.5, 10, 0.5)) + [10, 20, 50, 100]

# Variables to get the maximum accuracy and its lambda value
max_accuracy = -1
optimal_lambda = 0

# We run a loop over all the lambda values we want to test
for l in lambdas:
    lambd = l * np.eye(85)
    
    # Calculating the weight matrix of linear regression
    weights = np.linalg.inv(X.T@X + lambd)@X.T@y

    # Computing the unseen means by multiplying class attributes of unseen
    # with the weights
    means_unseen = class_attributes_unseen@weights

    # Predicting outputs (similar to convex.py)
    predictions = np.array([predict(means_unseen, test_point) for test_point in Xtest]).reshape(-1, 1)
    acc = accuracy_score(Ytest, predictions)
    accuracies.append(acc)
    if(acc>max_accuracy):
        max_accuracy = acc
        optimal_lambda = l

accuracies = np.array(accuracies)


# Plotting the accuracy vs lambda value plot on linear and log scale
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(lambdas, accuracies, label='Accuracy')
ax[0].axvline(x=optimal_lambda)
ax[0].set_title('Accuracy vs Lambda')
ax[0].set_xlabel('Lambda')
ax[0].set_ylabel('Accuracy score')

# Plotting on log scale in x-axis
ax[1].plot(lambdas, accuracies, label='Accuracy')
ax[1].axvline(x=optimal_lambda)
ax[1].set_title('Accuracy vs Lambda')
ax[1].set_xlabel('log(Lambda)')
ax[1].set_xscale('log')
ax[1].set_ylabel('Accuracy score')

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax[0].text(0.48, 0.8, r'optimal $\lambda=%.2f$'%optimal_lambda + '\n' + 'best accuracy=%.3f%%'%(max_accuracy*100), transform=ax[0].transAxes, fontsize=12, bbox=props)

plt.legend()
plt.show()

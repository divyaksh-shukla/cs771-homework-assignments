'''
# Method 1
**Steps**
1. Compute the mean of each seen class.
2. Compute the similarity (dot product based) of each unseen class with each of the seen classes.
3. Normalize the similarity vector (to that it sums to 1, since we are using a convex combination).
4. Compute the mean of each unseen class using a convex combination of means of seen classes.
5. Apply the model to predict labels on unseen class test inputs.
6. Compute classification accuracies.
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

# Computing the similarity between the seen and unseen classes by taking a dot 
# product between the vectors for each class.

# Here we are taking the matrix multiplication of `class_attributes_unseen` of 
# shape `(10, 85)` and `class_attributes_seen.T` of shape `(85, 40)`, which 
# gives us a matrix of shape `(10, 40)`, where each row contains the similarity
# vector of that unseen class with the seen classes.

# Computing similarity between seen and unseen classes
similarities = class_attributes_unseen@class_attributes_seen.T
print(f'Shape of similarities vector: {similarities.shape}')

# To compute the weighted sum of the means to find the means of the unseen 
# classes we have to normalize the above obtain similarity vectors.

# Normalizing the similarity vector to 1
sums = np.sum(similarities, axis=1)
norm_similarities = np.array([similarity/sum_val for similarity, sum_val in zip(similarities, sums)])
print(f'Shape of nomalized similarities vector: {norm_similarities.shape}')

# Now we estimate the means of the unseen classes by taking a weighted sum of 
# the `norm_similarities` with `means_seen`

# Computing means of unseen classes using convex combination of means of the seen classes.
means_unseen = norm_similarities@means_seen
print(f'Shape of unseen means: {means_unseen.shape}')

### Predict
# Generating predictions based on the unseen classes on `Xtest` and comparing 
# the same to `Ytest` to calculate accuracy

def predict(means, test_input):
    distances = np.array([(test_input - mean).T@(test_input - mean) for mean in means])
    return np.argmin(distances)+1.0

predictions = np.array([predict(means_unseen, test_point) for test_point in Xtest]).reshape(-1, 1)
print(f'Shape of predictions: {predictions.shape}')

# Computing the accuracy score
acc = accuracy_score(Ytest, predictions)
print(f'Accuracy = {acc*100:.3f}%')
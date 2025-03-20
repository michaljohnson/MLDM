import numpy as np
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Define the flip function
def flip(faces):
    faces = np.atleast_3d(faces)
    faces = faces.reshape(-1, 24, 24)  # convert faces from vectors to 2-d arrays
    faces = faces[:, :, ::-1]  # flip all the 2-d arrays left-to-right
    return faces.reshape(-1, 24**2)  # convert faces from 2-d arrays to vectors


# Load the dataset
trainingLabels = np.load("trainingLabels.npy")
trainingFaces = np.load("trainingFaces.npy")
testingLabels = np.load("testingLabels.npy")
testingFaces = np.load("testingFaces.npy")

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(trainingFaces, trainingLabels, test_size=0.2, random_state=42)

# Define the function to add Gaussian noise
def add_noise(faces, std_dev=0.1):
    noise = np.random.randn(*faces.shape) * std_dev
    return faces + noise

# Augment the training data
augmentedTrainingFaces = np.concatenate((X_train, flip(X_train), add_noise(X_train)))
augmentedTrainingLabels = np.concatenate((y_train, y_train, y_train))

# Hyperparameter tuning
best_val_accuracy = 0
best_C = None
C_values = [0.01, 0.1, 1, 10, 100]

for C in C_values:
    logisticRegressor = sklearn.linear_model.LogisticRegression(C=C, max_iter=500)
    logisticRegressor.fit(augmentedTrainingFaces, augmentedTrainingLabels)
    
    # Measure validation accuracy
    y_val_pred = logisticRegressor.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_C = C

# Train the final model with the best hyperparameter on the full augmented training set
logisticRegressor = sklearn.linear_model.LogisticRegression(C=best_C, max_iter=500)
logisticRegressor.fit(augmentedTrainingFaces, augmentedTrainingLabels)

# Measure testing accuracy
y_test_pred = logisticRegressor.predict(testingFaces)
test_accuracy = accuracy_score(testingLabels, y_test_pred)
print(f"Best C: {best_C}")
print(f"Test Accuracy: {test_accuracy}")

# Plot the results
trainingAccuracies = []
testingAccuracies = []
Mvalues = np.arange(100, 4001, 100)

for M in Mvalues:
    logisticRegressor = sklearn.linear_model.LogisticRegression(C=best_C, max_iter=500)
    X_train_subset = augmentedTrainingFaces[:M]
    y_train_subset = augmentedTrainingLabels[:M]
    logisticRegressor.fit(X_train_subset, y_train_subset)
    
    y_train_pred = logisticRegressor.predict(X_train_subset)
    training_accuracy = accuracy_score(y_train_subset, y_train_pred)
    trainingAccuracies.append(training_accuracy)
    
    y_test_pred = logisticRegressor.predict(testingFaces)
    testing_accuracy = accuracy_score(testingLabels, y_test_pred)
    testingAccuracies.append(testing_accuracy)

plt.plot(Mvalues, trainingAccuracies, label="Training")
plt.plot(Mvalues, testingAccuracies, label="Testing")
plt.xlabel("Training Set Size (M)")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Effect of Training Set Size on Training & Testing Accuracies with Data Augmentation and Hyperparameter Tuning")
plt.show()
# ML_LAB


**1. Mean median Mode**
import statistics

# Function to calculate mean
def calculate_mean(data):
    return sum(data) / len(data)

# Function to calculate median
def calculate_median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 0:
        middle1 = sorted_data[n // 2 - 1]
        middle2 = sorted_data[n // 2]
        return (middle1 + middle2) / 2
    else:
        return sorted_data[n // 2]

# Function to calculate mode
def calculate_mode(data):
    return statistics.mode(data)

# Function to calculate variance
def calculate_variance(data):
    mean_value = calculate_mean(data)
    squared_diff_sum = sum((x - mean_value) ** 2 for x in data)
    return squared_diff_sum / (len(data) - 1)

# Function to calculate standard deviation
def calculate_standard_deviation(data):
    variance_value = calculate_variance(data)
    return variance_value ** 0.5

# Example dataset
dataset = [10, 20, 30, 40, 50]

# Compute measures
mean_value = calculate_mean(dataset)
median_value = calculate_median(dataset)
mode_value = calculate_mode(dataset)
variance_value = calculate_variance(dataset)
std_deviation_value = calculate_standard_deviation(dataset)

# Print results
print(f"Dataset: {dataset}")
print(f"Mean: {mean_value:.2f}")
print(f"Median: {median_value:.2f}")
print(f"Mode: {mode_value}")
print(f"Variance: {variance_value:.2f}")
print(f"Standard Deviation: {std_deviation_value:.2f}")



**4. simple linear Regresions**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# To read data from Age_Income.csv file
dataFrame = pd.read_csv('Age_Income.csv')
# To place data in to age and income vectors
age = dataFrame['Age']
income = dataFrame['Income']

# number of points
size = np.size(age)
# To find the mean of age and income vector
mean_age = np.mean(age)
mean_income = np.mean(income)

# calculating cross-deviation and deviation about age
CD_ageincome = np.sum(income*age) - size*mean_income*mean_age
CD_ageage = np.sum(age*age) - size*mean_age*mean_age

# calculating regression coefficients
b1 = CD_ageincome / CD_ageage
b0 = mean_income - b1*mean_age
# to display coefficients
print("Estimated Coefficients :")
print("b0 = ",b0,"\nb1 = ",b1)
# To plot the actual points as scatter plot
plt.scatter(age, income, color = "b",marker = "o")
# TO predict response vector
response_Vec = b0 + b1*age
# To plot the regression line
plt.plot(age, response_Vec, color = "r")
# Placing labels
plt.xlabel('Age')
plt.ylabel('Income')
# To display plot
plt.show()



**7. KNN**
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the Iris dataset from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
df = pd.read_csv(url, names=names)

# Split features and labels
X = df.iloc[:, :-1].values   # Features
y = df.iloc[:, 4].values     # Target labels (class)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Feature scaling (important for KNN)   
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN model
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


**10. classification Algorithm**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer

# Step 1: Load the Dataset
data = load_breast_cancer()
print(data)
X, y = data.data, data.target


n_samples, n_features = X.shape
print(f"Number of samples: {n_samples}")
print(f"Number of features: {n_features}")

# --- Peek at the first few rows ---
df = pd.DataFrame(X, columns=data.feature_names)
df["target"] = y
df["target_label"] = pd.Series(y).map({i: name for i, name in enumerate(data.target_names)})

print("First 5 rows (features + target):")
print(pd.DataFrame(X, columns=data.feature_names).head())


# Step 2: Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Normalize the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train Classification Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

results = {}  # Store accuracy results

for name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Predict on test data
    accuracy = accuracy_score(y_test, y_pred)  # Evaluate accuracy
    results[name] = accuracy  # Store accuracy
    print(f"{name}:\n{classification_report(y_test, y_pred)}\n")  # Show classification report

# Step 5: Visualize Performance Comparison
df_results = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])

plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='Accuracy', data=df_results, palette='coolwarm')
plt.title("Performance Comparison of Classification Models")
plt.xlabel("Classification Model")
plt.ylabel("Accuracy Score")
plt.ylim(0, 1)
plt.show()

        

# Random Forest - Machine Learning Model Documentation

## 1. Introduction

A **Random Forest** is an ensemble learning algorithm that builds multiple decision trees and combines their predictions to improve accuracy and robustness. It is widely used for both classification and regression tasks. By aggregating the predictions of several trees, Random Forest mitigates the limitations of individual decision trees, such as overfitting and sensitivity to small changes in data.

---

## 2. Key Features

- **Ensemble Method**: Combines multiple decision trees to enhance performance.
- **Handles Non-linearity**: Captures complex relationships in data.
- **Works with Categorical and Numerical Data**: Supports both types of variables.
- **Robust to Noise and Overfitting**: Reduces variance by averaging predictions from multiple trees.
- **Feature Importance**: Provides insight into which features contribute the most to predictions.
- **Handles Missing Data Well**: Uses bootstrapping to improve predictions even with missing values.

---

## 3. Types of Random Forest Models

1. **Classification Random Forest**: Used when the target variable is categorical.
2. **Regression Random Forest**: Used when the target variable is continuous.

---

## 4. How Random Forest Works

Random Forest operates by constructing multiple decision trees and combining their results to make predictions. The steps involved are as follows:

1. **Bootstrapping**: Multiple subsets of the original dataset are created through random sampling with replacement.
2. **Decision Tree Training**: A decision tree is trained on each subset, using only a random subset of features at each split to enhance diversity.
3. **Aggregation of Predictions**:
   - For classification tasks, majority voting determines the final class label.
   - For regression tasks, the final prediction is the average of all tree predictions.

By combining multiple decision trees, Random Forest improves accuracy and reduces overfitting compared to individual decision trees.

---

## 5. Hyperparameters and Tuning

- **Number of Trees (`n_estimators`)**: The number of decision trees in the forest. More trees generally improve accuracy but increase computational cost.
- **Max Features (`max_features`)**: The number of features considered at each split. Lower values increase randomness and reduce correlation between trees.
- **Max Depth (`max_depth`)**: Limits the depth of individual trees to prevent overfitting.
- **Min Samples Split (`min_samples_split`)**: The minimum number of samples required to split a node.
- **Min Samples Leaf (`min_samples_leaf`)**: The minimum number of samples required in a leaf node to prevent overly complex trees.

---

## 6. When Random Forest Performs Better

Random Forest excels in the following scenarios:

- **When Overfitting is a Concern**: By averaging multiple trees, it reduces variance and generalizes better than a single decision tree.
- **When Dealing with High-Dimensional Data**: Random Forest selects important features automatically, making it effective for datasets with many variables.
- **When Feature Importance is Needed**: Provides insights into which features contribute most to predictions.
- **When Handling Noisy Data**: Less sensitive to noise and outliers due to bootstrapping and averaging.
- **When Working with Missing Data**: Can handle missing values effectively using bootstrapping and surrogate splits.

However, Random Forest may be slower and computationally expensive compared to simpler models, especially when dealing with very large datasets.

---

## 7. Implementing Random Forest in Python

### Classification Example:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

### Regression Example:
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_regressor.fit(X_train, y_train)

# Make predictions
y_pred = rf_regressor.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
```

---

## 8. Conclusion

Random Forest is a powerful ensemble learning algorithm that improves the performance of decision trees by reducing overfitting and increasing accuracy. It is widely used in real-world applications where robustness and interpretability are essential. While it requires more computational resources than single decision trees, its advantages make it a preferred choice in many scenarios.

---

## 9. References

- "Pattern Recognition and Machine Learning" – Christopher Bishop
- Scikit-learn Documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- "Introduction to Machine Learning with Python" – Andreas Müller and Sarah Guido

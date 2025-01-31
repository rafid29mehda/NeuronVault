# Decision Tree - Machine Learning Model Documentation

## 1. Introduction
A **Decision Tree** is a supervised learning algorithm used for classification and regression tasks. It models data using a tree-like structure, where decisions are made by splitting data at different nodes based on certain criteria. Decision trees are easy to understand and interpret, making them popular in various applications.

---

## 2. Key Features
- **Interpretable**: Easy to understand, visualize, and interpret.
- **Handles Non-linearity**: Captures complex relationships in data.
- **Works with Categorical and Numerical Data**: Supports both types of variables.
- **No Need for Feature Scaling**: Does not require standardization or normalization.
- **Handles Missing Values**: Can work with missing data to some extent.

---

## 3. Types of Decision Trees
1. **Classification Trees**: Used when the target variable is categorical.
2. **Regression Trees**: Used when the target variable is continuous.

---

## 4. Components of a Decision Tree
- **Root Node**: The starting point of the tree.
- **Internal Nodes**: Intermediate decision points.
- **Branches**: Represent the outcome of a decision.
- **Leaf Nodes**: Final decision points (class labels or continuous values).

---

## 5. How Decision Trees Work
1. **Select the Best Feature**: Choose a feature that provides the best split.
2. **Split the Data**: Partition the data into subsets based on selected feature.
3. **Repeat Process**: Continue splitting until a stopping condition is met.
4. **Make Predictions**: Assign final outputs at leaf nodes.

---

## 6. Splitting Criteria
### For Classification
- **Gini Index**: Measures impurity (lower is better).
- **Entropy (Information Gain)**: Measures information gain (higher is better).

### For Regression
- **Mean Squared Error (MSE)**: Measures variance reduction.
- **Mean Absolute Error (MAE)**: Measures absolute deviation.

---

## 7. Pruning Techniques
To prevent overfitting, pruning is applied:
- **Pre-Pruning (Early Stopping)**: Stops splitting based on a predefined criterion (e.g., minimum number of samples per node).
- **Post-Pruning (Reduced Error Pruning)**: Trims branches after training to simplify the model.

---

## 8. Advantages & Disadvantages
### Advantages:
- Simple and easy to interpret.
- Requires little data preprocessing.
- Handles both numerical and categorical data.
- Can capture interactions between features.

### Disadvantages:
- Prone to overfitting (solved by pruning or ensemble methods like Random Forests).
- Can be unstable with small changes in data.
- Less accurate compared to some advanced models like Gradient Boosting.

---

## 9. Implementing a Decision Tree in Python
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
dt_model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions
y_pred = dt_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

---

## 10. Use Cases
- **Medical Diagnosis**: Predicting diseases based on symptoms.
- **Customer Segmentation**: Categorizing customers based on behaviors.
- **Fraud Detection**: Identifying fraudulent transactions.
- **Stock Market Prediction**: Analyzing financial data for decision-making.

---

## 11. Enhancing Decision Trees
- **Ensemble Methods**: Random Forests, Gradient Boosting, AdaBoost improve performance.
- **Feature Engineering**: Selecting relevant features reduces overfitting.
- **Hyperparameter Tuning**: Adjusting max depth, minimum samples per split improves accuracy.

---

## 12. Conclusion
Decision Trees are a fundamental ML model, providing simple yet powerful classification and regression capabilities. While they have limitations like overfitting, these can be mitigated using ensemble techniques and pruning methods.

---

## 13. References
- "Pattern Recognition and Machine Learning" – Christopher Bishop
- Scikit-learn Documentation: https://scikit-learn.org/stable/
- "Introduction to Machine Learning with Python" – Andreas Müller and Sarah Guido


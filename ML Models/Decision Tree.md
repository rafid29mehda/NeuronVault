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

Decision Trees operate by recursively splitting data into subsets based on the most significant attribute at each step. The steps involved in constructing a decision tree are as follows:

1. **Feature Selection**: Identify the most relevant feature that maximizes the information gain or reduces impurity in classification tasks.
2. **Splitting the Data**: The dataset is divided based on the selected feature, forming multiple branches.
3. **Recursive Partitioning**: The process continues recursively for each subset, selecting the best feature at each level until a stopping condition is met (e.g., maximum tree depth, minimum samples per leaf, or impurity threshold).
4. **Stopping Criteria**: The recursive splitting stops when one of the following conditions is met:
   - All instances in a node belong to the same class.
   - A predefined maximum depth of the tree is reached.
   - The number of instances in a node is less than a minimum threshold.
5. **Prediction**: After training, new data points are classified by traversing the tree from the root node to a leaf node, following the conditions at each internal node.

The decision-making process can be represented as a series of if-else conditions, making it easily interpretable.

---

## 6. Splitting Criteria

### For Classification

When building a decision tree for classification tasks, we need a way to decide which attribute provides the best split at each node. The two main criteria used for classification are:

- **Gini Index**: The Gini Index, or Gini Impurity, measures how often a randomly chosen element from a set would be incorrectly labeled if randomly classified based on the distribution of labels. The formula for Gini Impurity is:
  
  \[ Gini = 1 - \sum_{i=1}^{n} p_i^2 \]
  
  where \( p_i \) is the probability of class \( i \). A lower Gini Index indicates a purer split, meaning the data is more homogeneous after the split.

- **Entropy (Information Gain)**: Entropy measures the amount of uncertainty or randomness in a dataset. It is calculated as:
  
  \[ Entropy = - \sum_{i=1}^{n} p_i \log_2 p_i \]
  
  where \( p_i \) is the probability of class \( i \). A lower entropy value indicates more homogeneity in a node. Information Gain is used to determine the attribute that provides the best split and is computed as:
  
  \[ IG = Entropy(Parent) - \sum_{j=1}^{k} \left( \frac{|S_j|}{|S|} \times Entropy(S_j) \right) \]
  
  where \( S_j \) represents each subset resulting from the split. A higher Information Gain means a better split.

### For Regression

For regression tasks, decision trees split data based on numerical target values. The following criteria are commonly used:

- **Mean Squared Error (MSE)**: This criterion minimizes the variance within each split by measuring the average squared difference between actual and predicted values. The formula for MSE is:
  
  \[ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y})^2 \]
  
  where \( y_i \) is the actual value and \( \hat{y} \) is the predicted mean value of the subset.

- **Mean Absolute Error (MAE)**: MAE measures the absolute differences between actual and predicted values to minimize overall deviation. The formula is:
  
  \[ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}| \]
  
  While MSE penalizes large errors more due to squaring, MAE treats all deviations equally.

These splitting criteria help decision trees effectively partition data and make accurate predictions for both classification and regression tasks.

---

## 7. Pruning Techniques

To prevent overfitting, pruning is applied:

- **Pre-Pruning (Early Stopping)**: This technique prevents excessive tree growth by stopping the tree's expansion at an early stage. It sets constraints like:
  - Minimum number of samples required to split a node.
  - Maximum depth of the tree.
  - Minimum impurity decrease required for a split.
  By stopping the tree before it becomes too complex, pre-pruning helps prevent overfitting but may risk underfitting if applied too aggressively.

- **Post-Pruning (Reduced Error Pruning)**: Unlike pre-pruning, post-pruning allows the tree to fully grow and then prunes unnecessary branches. It involves:
  - Removing branches that do not significantly contribute to predictive performance.
  - Evaluating the impact of pruning on validation data.
  - Reducing tree complexity while maintaining accuracy.
  This method helps optimize model performance by reducing overfitting while ensuring the tree captures significant patterns in the data.

---

## 8. Conclusion

Decision Trees are a fundamental ML model, providing simple yet powerful classification and regression capabilities. While they have limitations like overfitting, these can be mitigated using ensemble techniques and pruning methods.

---

## 9. References

- "Pattern Recognition and Machine Learning" – Christopher Bishop
- Scikit-learn Documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- "Introduction to Machine Learning with Python" – Andreas Müller and Sarah Guido


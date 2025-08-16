import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
# Replace with your dataset path
df = pd.read_csv("data/loan_data.csv")

# Features and target
X = df.drop(columns=["Default"])
y = df["Default"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall:", recall_score(y_test, y_pred_log))
print("F1 Score:", f1_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))

# Decision Tree (CART)
cart_model = DecisionTreeClassifier(random_state=42)
cart_model.fit(X_train, y_train)
y_pred_cart = cart_model.predict(X_test)

print("\n=== CART Decision Tree ===")
print("Accuracy:", accuracy_score(y_test, y_pred_cart))
print("Precision:", precision_score(y_test, y_pred_cart))
print("Recall:", recall_score(y_test, y_pred_cart))
print("F1 Score:", f1_score(y_test, y_pred_cart))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_cart))
